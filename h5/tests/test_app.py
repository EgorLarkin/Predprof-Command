import os
import sys
import pytest
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPreprocess:
    """Тесты модуля предобработки данных."""

    def test_extract_class_name_valid(self):
        """Тест извлечения имени из повреждённой метки."""
        from ml.preprocess import extract_class_name
        label = 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4Kepler-22b'
        assert extract_class_name(label) == 'Kepler-22b'

    def test_extract_class_name_gliese(self):
        """Тест извлечения для Gliese_ (короткое имя)."""
        from ml.preprocess import extract_class_name
        label = 'abcdef1234567890abcdef1234567890Gliese_'
        assert extract_class_name(label) == 'Gliese_'

    def test_extract_class_name_55cancri(self):
        """Тест извлечения для 55_Cancri_Bc."""
        from ml.preprocess import extract_class_name
        label = '0123456789abcdef0123456789abcdef55_Cancri_Bc'
        assert extract_class_name(label) == '55_Cancri_Bc'

    def test_name_to_label(self):
        """Тест преобразования имени в числовую метку."""
        from ml.preprocess import name_to_label
        assert name_to_label('55_Cancri_Bc') == 0
        assert name_to_label('Kepler-62f') == 19

    def test_label_to_name(self):
        """Тест преобразования числовой метки в имя."""
        from ml.preprocess import label_to_name
        assert label_to_name(0) == '55_Cancri_Bc'
        assert label_to_name(14) == 'Kepler-22b'

    def test_all_20_classes(self):
        """Проверка наличия 20 уникальных классов."""
        from config import Config
        assert len(Config.CLASS_NAMES) == 20
        assert len(set(Config.CLASS_NAMES)) == 20

    def test_load_data_shapes(self):
        """Тест формы данных после загрузки."""
        from ml.preprocess import load_and_preprocess_data
        from config import Config
        if not os.path.exists(Config.DATA_PATH):
            pytest.skip("Файл данных не найден")
        train_x, train_y, valid_x, valid_y, class_names = load_and_preprocess_data()
        assert train_x.shape == (1200, 80000, 1)
        assert valid_x.shape == (400, 80000, 1)
        assert train_y.shape == (1200,)
        assert valid_y.shape == (400,)
        assert len(class_names) == 20

    def test_labels_are_integers(self):
        """Тест что метки — целые числа от 0 до 19."""
        from ml.preprocess import load_and_preprocess_data
        from config import Config
        if not os.path.exists(Config.DATA_PATH):
            pytest.skip("Файл данных не найден")
        _, train_y, _, valid_y, _ = load_and_preprocess_data()
        assert train_y.dtype == np.int32
        assert valid_y.dtype == np.int32
        assert train_y.min() >= 0
        assert train_y.max() <= 19
        assert valid_y.min() >= 0
        assert valid_y.max() <= 19


class TestApp:
    """Тесты Flask-приложения."""

    @pytest.fixture
    def app(self):
        from app.routes import create_app
        app = create_app()
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False

        with app.app_context():
            from app.models import db, User
            db.create_all()
            admin = User(username='admin', first_name='Тест', last_name='Админ', role='admin')
            admin.set_password('admin')
            db.session.add(admin)
            db.session.commit()
            yield app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def login(self, client, username='admin', password='admin'):
        return client.post('/login', data={
            'username': username,
            'password': password
        }, follow_redirects=True)

    def test_login_page(self, client):
        """Тест отображения страницы входа."""
        resp = client.get('/login')
        assert resp.status_code == 200
        assert 'Вход в систему'.encode('utf-8') in resp.data

    def test_login_admin(self, client):
        """Тест авторизации администратора."""
        resp = self.login(client)
        assert resp.status_code == 200
        assert 'Панель администратора'.encode('utf-8') in resp.data

    def test_login_wrong_password(self, client):
        """Тест входа с неверным паролем."""
        resp = self.login(client, password='wrong')
        assert 'Неверный логин или пароль'.encode('utf-8') in resp.data

    def test_create_user(self, client, app):
        """Тест создания нового пользователя администратором."""
        self.login(client)
        resp = client.post('/admin/create_user', data={
            'first_name': 'Иван',
            'last_name': 'Петров',
            'username': 'ipetrov',
            'password': 'test123'
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert 'успешно создан'.encode('utf-8') in resp.data

        client.get('/logout')
        resp = self.login(client, 'ipetrov', 'test123')
        assert resp.status_code == 200

    def test_duplicate_user(self, client, app):
        """Тест создания дубликата пользователя."""
        self.login(client)
        client.post('/admin/create_user', data={
            'first_name': 'А', 'last_name': 'Б',
            'username': 'dup', 'password': '123'
        })
        resp = client.post('/admin/create_user', data={
            'first_name': 'В', 'last_name': 'Г',
            'username': 'dup', 'password': '456'
        }, follow_redirects=True)
        assert 'уже существует'.encode('utf-8') in resp.data

    def test_access_control(self, client, app):
        """Тест контроля доступа: пользователь не может войти в панель администратора."""
        self.login(client)
        client.post('/admin/create_user', data={
            'first_name': 'Test', 'last_name': 'User',
            'username': 'testuser', 'password': 'testpass'
        })
        client.get('/logout')

        self.login(client, 'testuser', 'testpass')
        resp = client.get('/admin', follow_redirects=True)
        assert 'Доступ запрещён'.encode('utf-8') in resp.data

    def test_logout(self, client):
        """Тест выхода из системы."""
        self.login(client)
        resp = client.get('/logout', follow_redirects=True)
        assert resp.status_code == 200

    def test_profile_page(self, client, app):
        """Тест страницы профиля пользователя."""
        self.login(client)
        client.post('/admin/create_user', data={
            'first_name': 'Мария', 'last_name': 'Иванова',
            'username': 'maria', 'password': 'pass'
        })
        client.get('/logout')
        self.login(client, 'maria', 'pass')
        resp = client.get('/profile')
        assert resp.status_code == 200
        assert 'Мария'.encode('utf-8') in resp.data

    def test_api_user_info(self, client, app):
        """Тест API информации о пользователе."""
        self.login(client)
        resp = client.get('/api/user_info')
        data = json.loads(resp.data)
        assert data['username'] == 'admin'
        assert data['role'] == 'admin'


class TestModel:
    """Тесты модели нейросети (если обучена)."""

    def test_model_build(self):
        """Тест создания архитектуры модели."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from ml.train import build_model
        model = build_model(input_shape=(80000, 1), num_classes=20)
        assert model.output_shape == (None, 20)
        assert model.input_shape == (None, 80000, 1)

    def test_model_file_exists(self):
        """Тест что файл модели существует."""
        from config import Config
        if not os.path.exists(Config.MODEL_PATH):
            pytest.skip("Модель ещё не обучена")
        assert os.path.getsize(Config.MODEL_PATH) > 0

    def test_model_prediction_shape(self):
        """Тест формы выхода модели."""
        from config import Config
        if not os.path.exists(Config.MODEL_PATH):
            pytest.skip("Модель ещё не обучена")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from ml.predict import predict
        dummy = np.random.randn(2, 80000, 1).astype(np.float32)
        result = predict(dummy)
        assert len(result['predictions']) == 2
        assert len(result['class_names']) == 2
        assert len(result['probabilities']) == 2
        assert len(result['probabilities'][0]) == 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
