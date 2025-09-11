#!/bin/bash

# Docker helper scripts for Vector DB API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Build production image
build_prod() {
    print_status "Building production Docker image..."
    check_docker
    docker build -t vector-db-api:latest .
    print_success "Production image built successfully!"
}

# Build development image (same as production but with dev environment)
build_dev() {
    print_status "Building development Docker image..."
    check_docker
    docker build --build-arg ENVIRONMENT=development -t vector-db-api:dev .
    print_success "Development image built successfully!"
}

# Run production container
run_prod() {
    print_status "Starting production container..."
    check_docker
    docker run -d \
        --name vector-db-api-prod \
        -p 8000:8000 \
        --restart unless-stopped \
        vector-db-api:latest
    print_success "Production container started! API available at http://localhost:8000"
}

# Run development container
run_dev() {
    print_status "Starting development container..."
    check_docker
    docker run -d \
        --name vector-db-api-dev \
        -p 8000:8000 \
        -e ENVIRONMENT=development \
        -v "$(pwd)/src:/app/src" \
        -v "$(pwd)/tests:/app/tests" \
        -v "$(pwd)/run_tests.py:/app/run_tests.py" \
        vector-db-api:dev \
        uvicorn vector_db_api.main:app --host 0.0.0.0 --port 8000 --reload
    print_success "Development container started! API available at http://localhost:8000"
}

# Run with docker-compose (production)
compose_up() {
    print_status "Starting services with docker-compose..."
    check_docker
    docker-compose up -d
    print_success "Services started! API available at http://localhost:8000"
}

# Run with docker-compose (development)
compose_up_dev() {
    print_status "Starting development services with docker-compose..."
    check_docker
    ENVIRONMENT=development docker-compose up -d
    print_success "Development services started! API available at http://localhost:8000"
}

# Stop containers
stop_containers() {
    print_status "Stopping containers..."
    docker-compose down 2>/dev/null || true
    docker stop vector-db-api-prod vector-db-api-dev 2>/dev/null || true
    docker rm vector-db-api-prod vector-db-api-dev 2>/dev/null || true
    print_success "Containers stopped and removed!"
}

# Run tests in container
test_container() {
    print_status "Running tests in container..."
    check_docker
    docker run --rm \
        -v "$(pwd)/src:/app/src" \
        -v "$(pwd)/tests:/app/tests" \
        -v "$(pwd)/run_tests.py:/app/run_tests.py" \
        vector-db-api:dev \
        python3 run_tests.py --verbose
}

# Show logs
show_logs() {
    local service=${1:-vector-db-api}
    print_status "Showing logs for $service..."
    docker-compose logs -f $service
}

# Clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    docker volume prune -f
    print_success "Cleanup completed!"
}

# Show container status
status() {
    print_status "Container status:"
    docker ps -a --filter "name=vector-db-api"
    echo ""
    print_status "Docker Compose services:"
    docker-compose ps 2>/dev/null || echo "No compose services running"
}

# Health check
health_check() {
    print_status "Checking API health..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy!"
    else
        print_error "API health check failed!"
        exit 1
    fi
}

# Show help
show_help() {
    echo "Vector DB API Docker Helper Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build-prod     Build production Docker image"
    echo "  build-dev      Build development Docker image"
    echo "  run-prod       Run production container"
    echo "  run-dev        Run development container"
    echo "  compose-up     Start services with docker-compose (production)"
    echo "  compose-up-dev Start services with docker-compose (development)"
    echo "  stop           Stop all containers"
    echo "  test           Run tests in container"
    echo "  logs [service] Show logs (default: vector-db-api)"
    echo "  status         Show container status"
    echo "  health         Check API health"
    echo "  cleanup        Clean up Docker resources"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build-prod"
    echo "  $0 compose-up"
    echo "  $0 logs nginx"
    echo "  $0 test"
}

# Main script logic
case "${1:-help}" in
    build-prod)
        build_prod
        ;;
    build-dev)
        build_dev
        ;;
    run-prod)
        run_prod
        ;;
    run-dev)
        run_dev
        ;;
    compose-up)
        compose_up
        ;;
    compose-up-dev)
        compose_up_dev
        ;;
    stop)
        stop_containers
        ;;
    test)
        test_container
        ;;
    logs)
        show_logs "$2"
        ;;
    status)
        status
        ;;
    health)
        health_check
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
