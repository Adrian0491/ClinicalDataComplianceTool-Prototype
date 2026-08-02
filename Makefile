# =============================================================================
# CDTool — Developer workflow
#
# Usage (Linux / macOS / Git Bash / WSL):
#   make up          start all Docker services
#   make migrate     run Alembic migrations against the running DB
#   make test        run the full integration test suite
#   make logs        tail FastAPI logs
#   make down        stop all services
#
# Windows (PowerShell) — run the equivalent commands from scripts/run.ps1
# =============================================================================

COMPOSE   = docker compose
BACKEND   = cd backend &&
PYTHONPATH_BACKEND = PYTHONPATH=..:.

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  CDTool developer commands"
	@echo ""
	@echo "  Infrastructure"
	@echo "    make up          Start all services (db, redis, elasticsearch, api, worker)"
	@echo "    make up-infra    Start only db, redis, elasticsearch (no app containers)"
	@echo "    make down        Stop all services"
	@echo "    make clean       Stop all services and remove volumes (DESTROYS DATA)"
	@echo "    make build       Rebuild Docker images"
	@echo "    make health      Check the API health endpoint"
	@echo "    make logs        Tail FastAPI logs"
	@echo "    make logs-all    Tail all service logs"
	@echo ""
	@echo "  Database"
	@echo "    make migrate     Run Alembic migrations (upgrade head)"
	@echo "    make migrate-down  Downgrade one migration step"
	@echo "    make makemigration msg=\"describe change\"  Autogenerate a new migration"
	@echo ""
	@echo "  Security"
	@echo "    make keys        Generate RSA-2048 JWT key pair → .env"
	@echo ""
	@echo "  Frontend"
	@echo "    make frontend    Start Angular dev server (proxy → localhost:8000)"
	@echo "    make frontend-build  Production build to frontend/dist/"
	@echo ""
	@echo "  Testing"
	@echo "    make test        Run the full integration test suite"
	@echo "    make test-fast   Run tests, stop on first failure"
	@echo "    make test-auth   Run only auth tests"
	@echo ""

# ── Infrastructure ────────────────────────────────────────────────────────────
.PHONY: up
up:
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  Services starting. Wait ~30 s for Elasticsearch, then run: make migrate"
	@echo "  API:            http://localhost:8000"
	@echo "  API docs (dev): http://localhost:8000/docs"
	@echo "  PostgreSQL:     localhost:5432"
	@echo "  Elasticsearch:  http://localhost:9200"
	@echo ""

.PHONY: up-infra
up-infra:
	$(COMPOSE) up -d db db-init redis elasticsearch
	@echo "  Infra only started. Run 'make migrate' once db is healthy."

.PHONY: down
down:
	$(COMPOSE) down

.PHONY: clean
clean:
	$(COMPOSE) down -v
	@echo "  All services stopped and volumes removed."

.PHONY: build
build:
	$(COMPOSE) build --no-cache

.PHONY: health
health:
	@curl -sf http://localhost:8000/health | python -m json.tool || \
		echo "API not reachable — is 'make up' running?"

.PHONY: logs
logs:
	$(COMPOSE) logs -f api

.PHONY: logs-all
logs-all:
	$(COMPOSE) logs -f

# ── Database ──────────────────────────────────────────────────────────────────
.PHONY: migrate
migrate:
	$(COMPOSE) exec api sh -c \
		"cd /app/backend && alembic upgrade head"

.PHONY: migrate-down
migrate-down:
	$(COMPOSE) exec api sh -c \
		"cd /app/backend && alembic downgrade -1"

# Usage: make makemigration msg="add user preferences"
.PHONY: makemigration
makemigration:
	$(COMPOSE) exec api sh -c \
		"cd /app/backend && alembic revision --autogenerate -m '$(msg)'"

# ── Security ──────────────────────────────────────────────────────────────────
.PHONY: keys
keys:
	python backend/scripts/generate_keys.py
	@echo ""
	@echo "  Keys written to .env"
	@echo "  Restart services: make down && make up && make migrate"
	@echo ""

# ── Frontend ──────────────────────────────────────────────────────────────────
FRONTEND = cd frontend &&

.PHONY: frontend
frontend:
	$(FRONTEND) npx ng serve --open

.PHONY: frontend-build
frontend-build:
	$(FRONTEND) npx ng build --configuration production
	@echo ""
	@echo "  Production build written to frontend/dist/"
	@echo ""

# ── Testing ───────────────────────────────────────────────────────────────────
# Tests run against a real PostgreSQL DB (cdtool_test).
# The db + db-init services must be running before 'make test'.
.PHONY: test
test:
	$(BACKEND) $(PYTHONPATH_BACKEND) pytest tests/ -v

.PHONY: test-fast
test-fast:
	$(BACKEND) $(PYTHONPATH_BACKEND) pytest tests/ -v -x

.PHONY: test-auth
test-auth:
	$(BACKEND) $(PYTHONPATH_BACKEND) pytest tests/test_auth.py -v

.PHONY: test-studies
test-studies:
	$(BACKEND) $(PYTHONPATH_BACKEND) pytest tests/test_studies.py -v

.PHONY: test-validation
test-validation:
	$(BACKEND) $(PYTHONPATH_BACKEND) pytest tests/test_validation.py -v

.PHONY: test-findings
test-findings:
	$(BACKEND) $(PYTHONPATH_BACKEND) pytest tests/test_findings.py -v

# app/validators + app/ingest are pure pandas-in/pandas-out — no DB needed,
# so this is kept separate from 'make test' rather than folded into the
# slower Postgres-backed integration suite.
.PHONY: test-validators
test-validators:
	$(BACKEND) $(PYTHONPATH_BACKEND) pytest app/validators/ app/ingest/ -v
