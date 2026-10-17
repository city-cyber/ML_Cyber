# Упаковка и запуск антиспам-модели

Этап 3: модель упакована в FastAPI-сервис и Docker-образ `project-antispam:latest`.
Публичное размещение (этап 4) пока не выполняется.

## Быстрый запуск на Windows

Откройте PowerShell в папке `PROJECT_antispam`, запустите Docker Desktop
в режиме Linux-контейнеров и выполните:

```powershell
.\start_docker.cmd
```

Обёртка `.cmd` запускает `start_docker.ps1` с разовым разрешением выполнения,
без изменения системной политики Windows. Сам PowerShell-скрипт проверяет доступность Docker,
при отсутствии модели запускает её экспорт,
собирает образ, поднимает контейнер через Compose и ждёт успешного healthcheck.
Затем он сверяет SHA-256 модели в контейнере с локальным артефактом.
При любой ошибке возвращается ненулевой код; успешная готовность не имитируется.

Если PowerShell запрещает выполнение локальных скриптов, используйте разовый запуск:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_docker.ps1
```

После готовности:

- Swagger и форма проверки писем: http://127.0.0.1:8000/docs
- Healthcheck: http://127.0.0.1:8000/health
- Информация о модели: http://127.0.0.1:8000/model

Варианты запуска:

```powershell
.\start_docker.cmd -SkipBuild          # готовый образ, без повторной сборки
.\start_docker.cmd -Port 8080          # слушать http://127.0.0.1:8080
.\start_docker.cmd -ExportModel        # восстановить модель и включить её в образ
.\stop_docker.cmd                      # остановить только Compose-проект antispam
```

При изменении локальной модели не используйте `-SkipBuild`: модель включается в образ
во время сборки. Скрипты вычисляют пути относительно себя, поэтому могут вызываться
из другой рабочей папки.

## Что сохранено в артефакте

`microservice/app/model/model.joblib` содержит обученный TF-IDF + классификатор,
выбранный порог, имя модели и тип оценки. `metadata.json` содержит SHA-256,
версии библиотек, число обучающих строк, отпечаток train и исходные метрики эксперимента.

При старте сервис проверяет целостность, версии scikit-learn/numpy/scipy/joblib,
версию предобработки и согласованность порога. Без корректной модели приложение
не запускается. Joblib-файлы принимаются только из собственного каталога модели,
загрузить модель через HTTP нельзя.

Пересоздать артефакт из сохранённых результатов без новой кросс-валидации:

```powershell
.\.venv\Scripts\python.exe export_model.py
```

Используется уже выбранная конфигурация и сохранённый train. Сравниваются оценки
и решения на validation, test и очищенном внешнем контроле с исходными CSV.
Если результаты различаются, экспорт завершается ошибкой. Отчёт:
`reports/export_verification.json`.

Для полного повторения обучения:

```powershell
.\.venv\Scripts\python.exe build.py
```

Теперь `build.py` автоматически сохраняет модель для сервиса. Параметр `--no-export`
позволяет выполнить только ML-конвейер.

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Модель загружена и сервис готов |
| GET | `/model` | Имя модели, порог, SHA-256 и версии |
| POST | `/predict` | JSON с текстом или темой и телом |
| POST | `/predict/eml` | `.eml` через multipart/form-data |
| GET | `/docs` | Swagger UI |
| GET | `/openapi.json` | Машиночитаемое описание API |

В `/predict` передаётся **либо** `text`, **либо** `subject` и `body`.
Для HTML-тела дополнительно укажите `body_format: "html"`.
Текст нормализуется тем же кодом, что использовался при обучении.

```powershell
$payload = @{
    subject = 'Project meeting'
    body = 'Please review the project report before Monday.'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
    -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

Можно отправить подготовленные примеры через `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/predict.json"
curl.exe -X POST http://127.0.0.1:8000/predict/eml -F "file=@examples/ham.eml;type=message/rfc822"
```

Ответ:

```json
{
  "label": "ham",
  "is_spam": false,
  "score": 0.12,
  "threshold": 0.8629806122399415,
  "score_kind": "probability_estimate",
  "model_name": "ComplementNB alpha=1.0"
}
```

Значение `score` в примере условное; сервис возвращает фактическую оценку.
Порог фиксирован по исходной validation; клиент не может менять его в запросе.
Оценка не является отдельно откалиброванной вероятностью.

Пустые и некорректные запросы получают HTTP 422. Максимальная длина JSON-текста —
200 000 символов; для `.eml` предел — 2 МиБ, превышение возвращает HTTP 413.
Вложения не исполняются и не включаются в признаки; используются тема и видимый текст тела.
Ни обучения, ни обращения к исходным корпусам при HTTP-запросах нет.

## Docker и Compose вручную

Из корня проекта (PowerShell, Linux или macOS):

```text
docker compose build antispam
docker compose up -d --no-build --wait --wait-timeout 120 antispam
docker compose ps
docker compose logs --tail 100 antispam
docker compose down
```

Сервис доступен только на localhost; порт можно изменить через `ANTISPAM_PORT`.
В контейнере Uvicorn слушает `0.0.0.0:8000`. Модель включена в образ;
папки данных, notebook, `.venv`, тесты и отчёты исключены из build context.
Запуск идёт от UID/GID 10001, с файловой системой только для чтения и временной `/tmp`.
Перезапуск — `unless-stopped`; healthcheck проверяет `/health`.

Образ собирается для архитектуры Docker Engine. В текущем окружении это Linux ARM64.
Для другого компьютера можно пересобрать тот же Dockerfile на его архитектуре.
Публикация в Docker Hub и загрузка на хостинг не выполнялись.

## Проверка

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/smoke_api.py
```

HTTP-проверка сравнивает реальные ответы контейнера с сохранёнными предсказаниями
для ham, найденного и пропущенного spam, проверяет загрузку `.eml`, healthcheck,
SHA-256 модели и отклонение пустого текста. Результат:
`reports/docker_api_verification.json`.

Для слайда об упаковке используйте скриншот Swagger, ответ `/health`, сведения
`/model` и этот отчёт. Обучение и контрольные метрики подробно разобраны в основном ноутбуке.

Источники реализации: [FastAPI Docker](https://fastapi.tiangolo.com/deployment/docker/),
[Dockerfile](https://docs.docker.com/reference/dockerfile/),
[сохранение моделей scikit-learn](https://scikit-learn.org/stable/model_persistence.html).
