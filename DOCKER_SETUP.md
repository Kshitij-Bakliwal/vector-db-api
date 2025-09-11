# Docker Setup

Simple Docker setup for the Vector DB API with a Makefile for easy commands.

## 🚀 Quick Start

```bash
# Build and run production
make setup

# Build and run development (with hot reloading)
make dev-setup

# Check what's running
make status

# Check API health
make health
```

## 📋 Available Commands

Run `make help` to see all available commands:

```bash
make help
```

### Key Commands

| Command | Description |
|---------|-------------|
| `make build` | Build production Docker image |
| `make run` | Run production container |
| `make dev` | Run development container with hot reloading |
| `make stop` | Stop all containers |
| `make test` | Run tests in container |
| `make health` | Check API health |
| `make status` | Show container status |
| `make logs` | Show container logs |
| `make clean` | Clean up Docker resources |

### Development Workflow

```bash
# Start development environment
make dev-setup

# Make changes to src/ files - they will auto-reload
# Check logs if needed
make logs

# Run tests
make test

# Stop when done
make stop
```

### Production Workflow

```bash
# Build and run production
make setup

# Check status
make status

# Check health
make health

# Stop when done
make stop
```

## 🌐 Access Points

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

## 🔧 Files

- `Dockerfile` - Unified container for dev/prod
- `docker-compose.yml` - Simple deployment
- `Makefile` - Build and run commands
- `scripts/docker-scripts.sh` - Alternative helper scripts
