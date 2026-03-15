/**
 * AlienSignal AI — JavaScript для дашборда (графики и загрузка тестовых данных)
 */

// Цветовая палитра для графиков
const COLORS = [
    '#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe',
    '#00f2fe', '#43e97b', '#fa709a', '#fee140', '#30cfd0',
    '#a18cd1', '#fbc2eb', '#ff9a9e', '#fad0c4', '#ffecd2',
    '#fcb69f', '#a1c4fd', '#c2e9fb', '#d4fc79', '#96e6a1'
];

// Настройки zoom-плагина (масштабирование)
const zoomOptions = {
    zoom: {
        wheel: { enabled: true },
        pinch: { enabled: true },
        mode: 'xy',
    },
    pan: {
        enabled: true,
        mode: 'xy',
    },
};

// Хранение экземпляров графиков для обновления
let charts = {};

// ==================== ИНИЦИАЛИЗАЦИЯ ====================

document.addEventListener('DOMContentLoaded', () => {
    loadTrainingHistory();
    loadDatasetInfo();
    setupUploadForm();
});

// ==================== ЗАГРУЗКА ДАННЫХ ====================

async function loadTrainingHistory() {
    try {
        const resp = await fetch('/api/training_history');
        if (!resp.ok) return;
        const data = await resp.json();
        renderAccuracyChart(data);
        renderLossChart(data);
    } catch (e) {
        console.error('Ошибка загрузки истории обучения:', e);
    }
}

async function loadDatasetInfo() {
    try {
        const resp = await fetch('/api/dataset_info');
        if (!resp.ok) return;
        const data = await resp.json();
        renderTrainDistChart(data);
        renderTop5ValidChart(data);
    } catch (e) {
        console.error('Ошибка загрузки информации о данных:', e);
    }
}

// ==================== ГРАФИКИ ====================

/**
 * 1. График зависимости точности от эпох обучения
 */
function renderAccuracyChart(history) {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;

    const epochs = history.accuracy.map((_, i) => i + 1);

    if (charts.accuracy) charts.accuracy.destroy();
    charts.accuracy = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Точность (обучение)',
                    data: history.accuracy,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102,126,234,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Точность (валидация)',
                    data: history.val_accuracy,
                    borderColor: '#f5576c',
                    backgroundColor: 'rgba(245,87,108,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                zoom: zoomOptions,
                title: { display: false }
            },
            scales: {
                x: { title: { display: true, text: 'Эпоха' } },
                y: { title: { display: true, text: 'Точность' }, min: 0, max: 1 }
            },
            ondblclick: null,
        }
    });

    // Двойной клик — сброс масштаба
    ctx.ondblclick = () => charts.accuracy.resetZoom();
}

/**
 * 2. Диаграмма распределения записей по цивилизациям (обучение)
 */
function renderTrainDistChart(info) {
    const ctx = document.getElementById('trainDistChart');
    if (!ctx) return;

    const labels = Object.keys(info.train_distribution);
    const values = Object.values(info.train_distribution);

    if (charts.trainDist) charts.trainDist.destroy();
    charts.trainDist = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Количество записей',
                data: values,
                backgroundColor: COLORS.slice(0, labels.length),
                borderWidth: 1,
                borderColor: '#fff',
            }]
        },
        options: {
            responsive: true,
            plugins: {
                zoom: zoomOptions,
                legend: { display: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Цивилизация' },
                    ticks: { maxRotation: 45, minRotation: 45, font: { size: 10 } }
                },
                y: { title: { display: true, text: 'Количество' }, beginAtZero: true }
            }
        }
    });

    ctx.ondblclick = () => charts.trainDist.resetZoom();
}

/**
 * 3. Точность определения каждой записи из тестового набора
 */
function renderPerSampleChart(testResult) {
    const ctx = document.getElementById('perSampleChart');
    if (!ctx) return;

    const placeholder = document.getElementById('perSamplePlaceholder');
    if (placeholder) placeholder.style.display = 'none';

    const labels = testResult.per_sample_confidence.map((_, i) => `#${i + 1}`);
    const colors = testResult.per_sample_accuracy
        ? testResult.per_sample_accuracy.map(v => v === 1 ? '#43e97b' : '#f5576c')
        : testResult.per_sample_confidence.map(() => '#4facfe');

    if (charts.perSample) charts.perSample.destroy();
    charts.perSample = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Уверенность предсказания',
                data: testResult.per_sample_confidence,
                backgroundColor: colors,
                borderWidth: 0,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                zoom: zoomOptions,
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const idx = context.dataIndex;
                            let lines = [];
                            if (testResult.predicted_names) {
                                lines.push('Предсказание: ' + testResult.predicted_names[idx]);
                            }
                            if (testResult.true_names) {
                                lines.push('Истинный: ' + testResult.true_names[idx]);
                            }
                            if (testResult.per_sample_accuracy) {
                                lines.push(testResult.per_sample_accuracy[idx] ? '✓ Верно' : '✗ Ошибка');
                            }
                            return lines;
                        }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Запись' }, ticks: { font: { size: 8 } } },
                y: { title: { display: true, text: 'Уверенность' }, min: 0, max: 1 }
            }
        }
    });

    ctx.ondblclick = () => charts.perSample.resetZoom();
}

/**
 * 4. Топ-5 наиболее часто встречающихся классов в валидационном наборе
 */
function renderTop5ValidChart(info) {
    const ctx = document.getElementById('top5ValidChart');
    if (!ctx) return;

    // Сортируем по количеству и берём топ-5
    const entries = Object.entries(info.valid_distribution)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    const labels = entries.map(e => e[0]);
    const values = entries.map(e => e[1]);

    if (charts.top5Valid) charts.top5Valid.destroy();
    charts.top5Valid = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: ['#667eea', '#f5576c', '#43e97b', '#fa709a', '#4facfe'],
                borderWidth: 2,
                borderColor: '#fff',
            }]
        },
        options: {
            responsive: true,
            plugins: {
                zoom: zoomOptions,
                legend: { position: 'bottom' },
                title: {
                    display: true,
                    text: 'Топ-5 классов (валидация)',
                    font: { size: 14 }
                }
            }
        }
    });
}

/**
 * 5. График потерь (Loss) по эпохам
 */
function renderLossChart(history) {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;

    const epochs = history.loss.map((_, i) => i + 1);

    if (charts.loss) charts.loss.destroy();
    charts.loss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Потери (обучение)',
                    data: history.loss,
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118,75,162,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Потери (валидация)',
                    data: history.val_loss,
                    borderColor: '#f093fb',
                    backgroundColor: 'rgba(240,147,251,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                zoom: zoomOptions,
            },
            scales: {
                x: { title: { display: true, text: 'Эпоха' } },
                y: { title: { display: true, text: 'Loss' }, min: 0 }
            }
        }
    });

    ctx.ondblclick = () => charts.loss.resetZoom();
}

// ==================== ЗАГРУЗКА ФАЙЛА ====================

function setupUploadForm() {
    const form = document.getElementById('uploadForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const fileInput = document.getElementById('testFile');
        const btn = document.getElementById('uploadBtn');
        const progress = document.getElementById('uploadProgress');
        const resultBody = document.getElementById('testResultBody');

        if (!fileInput.files.length) {
            alert('Выберите файл');
            return;
        }

        // Показать прогресс
        btn.disabled = true;
        progress.classList.remove('d-none');
        resultBody.innerHTML = '<p class="text-center loading"><i class="fas fa-spinner fa-spin me-2"></i>Загрузка и оценка модели...</p>';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const resp = await fetch('/api/upload_test', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || 'Ошибка сервера');
            }

            // Отображение результатов
            displayTestResults(data, resultBody);

            // Обновление графика per-sample
            if (data.per_sample_confidence) {
                renderPerSampleChart(data);
            }

        } catch (err) {
            resultBody.innerHTML = `
                <div class="alert alert-danger mb-0">
                    <i class="fas fa-exclamation-triangle me-2"></i>${err.message}
                </div>`;
        } finally {
            btn.disabled = false;
            progress.classList.add('d-none');
        }
    });
}

function displayTestResults(data, container) {
    let html = '<div class="row">';

    if (data.accuracy !== null && data.accuracy !== undefined) {
        html += `
            <div class="col-6">
                <div class="stat-card">
                    <div class="stat-value text-success">${(data.accuracy * 100).toFixed(2)}%</div>
                    <div class="stat-label">Точность (Accuracy)</div>
                </div>
            </div>`;
    }

    if (data.loss !== null && data.loss !== undefined) {
        html += `
            <div class="col-6">
                <div class="stat-card">
                    <div class="stat-value text-danger">${data.loss.toFixed(4)}</div>
                    <div class="stat-label">Потери (Loss)</div>
                </div>
            </div>`;
    }

    html += `
        <div class="col-12 mt-2">
            <div class="stat-card">
                <div class="stat-value text-primary">${data.total_samples}</div>
                <div class="stat-label">Всего записей</div>
            </div>
        </div>
    </div>`;

    container.innerHTML = html;
}
