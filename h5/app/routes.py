import os
import json
import numpy as np
import tempfile
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, session
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)
from werkzeug.utils import secure_filename

from app.models import db, User
from config import Config


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static')
    )
    app.config.from_object(Config)

    db.init_app(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @app.route('/')
    def index():
        if current_user.is_authenticated:
            if current_user.is_admin:
                return redirect(url_for('admin_panel'))
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('index'))

        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')

            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                flash('Вход выполнен успешно!', 'success')
                if user.is_admin:
                    return redirect(url_for('admin_panel'))
                return redirect(url_for('dashboard'))
            else:
                flash('Неверный логин или пароль.', 'error')

        return render_template('login.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('Вы вышли из системы.', 'info')
        return redirect(url_for('login'))

    @app.route('/admin')
    @login_required
    def admin_panel():
        if not current_user.is_admin:
            flash('Доступ запрещён.', 'error')
            return redirect(url_for('dashboard'))
        users = User.query.filter_by(role='user').all()
        return render_template('admin.html', users=users)

    @app.route('/admin/create_user', methods=['POST'])
    @login_required
    def create_user():
        if not current_user.is_admin:
            return jsonify({'error': 'Доступ запрещён'}), 403

        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not all([first_name, last_name, username, password]):
            flash('Все поля обязательны для заполнения.', 'error')
            return redirect(url_for('admin_panel'))

        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким логином уже существует.', 'error')
            return redirect(url_for('admin_panel'))

        user = User(
            username=username,
            first_name=first_name,
            last_name=last_name,
            role='user'
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash(f'Пользователь {username} успешно создан!', 'success')
        return redirect(url_for('admin_panel'))

    @app.route('/dashboard')
    @login_required
    def dashboard():
        if current_user.is_admin:
            return redirect(url_for('admin_panel'))
        return render_template('dashboard.html')

    @app.route('/profile')
    @login_required
    def profile():
        return render_template('profile.html')

    @app.route('/api/user_info')
    @login_required
    def api_user_info():
        return jsonify(current_user.to_dict())

    @app.route('/api/training_history')
    @login_required
    def api_training_history():
        """Возвращает историю обучения модели."""
        history_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'training_history.json')
        if not os.path.exists(history_path):
            return jsonify({'error': 'История обучения не найдена'}), 404
        with open(history_path, 'r') as f:
            history = json.load(f)
        return jsonify(history)

    @app.route('/api/dataset_info')
    @login_required
    def api_dataset_info():
        """Возвращает информацию о наборе данных."""
        info_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'dataset_info.json')
        if not os.path.exists(info_path):
            return jsonify({'error': 'Информация о данных не найдена'}), 404
        with open(info_path, 'r') as f:
            info = json.load(f)
        return jsonify(info)

    @app.route('/api/upload_test', methods=['POST'])
    @login_required
    def api_upload_test():
        """Загрузка и оценка тестового набора данных."""
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не выбран'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        if not file.filename.endswith('.npz'):
            return jsonify({'error': 'Поддерживается только формат .npz'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            from ml.preprocess import load_test_data
            from ml.predict import evaluate, predict

            test_x, test_y = load_test_data(filepath)

            if test_y is not None:
                result = evaluate(test_x, test_y)
            else:
                result = predict(test_x)
                result['loss'] = None
                result['accuracy'] = None

            result['total_samples'] = len(test_x)

            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'last_test_result.json')

            save_result = {k: v for k, v in result.items() if k != 'probabilities'}
            with open(result_path, 'w') as f:
                json.dump(save_result, f, indent=2, ensure_ascii=False)

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    @app.route('/api/last_test_result')
    @login_required
    def api_last_test_result():
        """Возвращает последний результат тестирования."""
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'last_test_result.json')
        if not os.path.exists(result_path):
            return jsonify({'error': 'Нет результатов тестирования'}), 404
        with open(result_path, 'r') as f:
            result = json.load(f)
        return jsonify(result)

    with app.app_context():
        db.create_all()

        if not User.query.filter_by(role='admin').first():
            admin = User(
                username='admin',
                first_name='Михаил',
                last_name='Администратор',
                role='admin'
            )
            admin.set_password('admin')
            db.session.add(admin)
            db.session.commit()
            print("Создан администратор: admin / admin")

    return app
