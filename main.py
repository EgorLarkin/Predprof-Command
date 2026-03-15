from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_change_me'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

DB_FILE = 'database.db'

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not cursor.fetchone():
            hash_pw = generate_password_hash('admin123')
            cursor.execute('INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)', ('admin', hash_pw, 1))
        conn.commit()

class User(UserMixin):
    def __init__(self, id, username, is_admin=False):
        self.id = id
        self.username = username
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, is_admin FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        if user_data:
            return User(user_data[0], user_data[1], bool(user_data[2]))
    return None

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, username, password_hash, is_admin FROM users WHERE username = ?', (username,))
            user_data = cursor.fetchone()
            
            if user_data and check_password_hash(user_data[2], password):
                user = User(user_data[0], user_data[1], bool(user_data[3]))
                login_user(user)
                return redirect(url_for('dashboard'))
        
        flash('Неверный логин или пароль', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', username=current_user.username, is_admin=current_user.is_admin)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash('Доступ запрещен', 'error')
        return redirect(url_for('dashboard'))
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, is_admin FROM users WHERE username != "admin"')
        users = cursor.fetchall()
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add':
            new_user = request.form.get('new_username')
            new_pass = request.form.get('new_password')
            if new_user and new_pass:
                try:
                    pwd_hash = generate_password_hash(new_pass)
                    with sqlite3.connect(DB_FILE) as conn:
                        cursor = conn.cursor()
                        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (new_user, pwd_hash))
                        conn.commit()
                    flash(f'Пользователь {new_user} добавлен', 'success')
                except sqlite3.IntegrityError:
                    flash('Такой пользователь уже существует', 'error')
        
        elif action == 'delete':
            user_id = request.form.get('user_id')
            if user_id:
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
                    conn.commit()
                flash('Пользователь удален', 'success')

        elif action == 'toggle_admin':
            user_id = request.form.get('user_id')
            if user_id:
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT is_admin FROM users WHERE id = ?', (user_id,))
                    res = cursor.fetchone()
                    if res:
                        new_status = 0 if res[0] else 1
                        cursor.execute('UPDATE users SET is_admin = ? WHERE id = ?', (new_status, user_id))
                        conn.commit()
                        status_text = "назначен админом" if new_status else "разжалован"
                        flash(f'Пользователь {status_text}', 'success')
        
        return redirect(url_for('admin_panel'))

    return render_template('admin.html', users=users)

def process_audio_mock(file_stream):
    return {
        "labels": ["Инопланетный сигнал", "Шум космоса", "Радиопомехи", "Тишина"],
        "probabilities": [0.0, 0.0, 0.0, 0.0],
        "frequencies": list(range(0, 100, 5)),
        "intensities": [0.0] * 20
    }

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Файл не выбран'}), 400

    analytics_data = process_audio_mock(file.stream)

    return jsonify({
        'success': True,
        'filename': file.filename,
        'analytics': analytics_data
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True)