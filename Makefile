# Vector DB API - Docker Makefile
# Simple commands for building and running the application

.PHONY: help build run stop clean test health logs status dev dev-stop

# Default target
help: ## Show this help message
	@echo "Vector DB API - Docker Commands"
	@echo "================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make build    # Build production image"
	@echo "  make run      # Run production container"
	@echo "  make dev      # Run development container with hot reloading"
	@echo "  make test     # Run tests in container"

# Build production image
build: ## Build production Docker image
	@echo "🔨 Building production image..."
	docker build -t vector-db-api:latest .
	@echo "✅ Production image built successfully!"

# Build development image
build-dev: ## Build development Docker image
	@echo "🔨 Building development image..."
	docker build --build-arg ENVIRONMENT=development -t vector-db-api:dev .
	@echo "✅ Development image built successfully!"

# Run production container
run: ## Run production container
	@echo "🚀 Starting production container..."
	docker run -d \
		--name vector-db-api \
		-p 8000:8000 \
		--restart unless-stopped \
		vector-db-api:latest
	@echo "✅ Production container started!"
	@echo "🌐 API available at: http://localhost:8000"
	@echo "📚 API docs at: http://localhost:8000/docs"

# Run development container with hot reloading
dev: ## Run development container with hot reloading
	@echo "🚀 Starting development container..."
	docker run -d \
		--name vector-db-api-dev \
		-p 8000:8000 \
		-e ENVIRONMENT=development \
		-v "$(PWD)/src:/app/src" \
		-v "$(PWD)/tests:/app/tests" \
		-v "$(PWD)/run_tests.py:/app/run_tests.py" \
		vector-db-api:dev \
		uvicorn vector_db_api.main:app --host 0.0.0.0 --port 8000 --reload
	@echo "✅ Development container started!"
	@echo "🌐 API available at: http://localhost:8000"
	@echo "🔄 Hot reloading enabled - changes to src/ will auto-reload"

# Run with docker-compose (production)
compose: ## Run with docker-compose (production)
	@echo "🚀 Starting services with docker-compose..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "🌐 API available at: http://localhost:8000"

# Run with docker-compose (development)
compose-dev: ## Run with docker-compose (development)
	@echo "🚀 Starting development services with docker-compose..."
	ENVIRONMENT=development docker-compose up -d
	@echo "✅ Development services started!"
	@echo "🌐 API available at: http://localhost:8000"

# Stop containers
stop: ## Stop all containers
	@echo "🛑 Stopping containers..."
	@docker-compose down 2>/dev/null || true
	@docker stop vector-db-api vector-db-api-dev 2>/dev/null || true
	@docker rm vector-db-api vector-db-api-dev 2>/dev/null || true
	@echo "✅ Containers stopped and removed!"

# Stop development container
dev-stop: ## Stop development container
	@echo "🛑 Stopping development container..."
	@docker stop vector-db-api-dev 2>/dev/null || true
	@docker rm vector-db-api-dev 2>/dev/null || true
	@echo "✅ Development container stopped!"

# Run tests in container
test: ## Run tests in container
	@echo "🧪 Running tests in container..."
	docker run --rm \
		-v "$(PWD)/src:/app/src" \
		-v "$(PWD)/tests:/app/tests" \
		-v "$(PWD)/run_tests.py:/app/run_tests.py" \
		vector-db-api:dev \
		python3 run_tests.py --verbose
	@echo "✅ Tests completed!"

# Check API health
health: ## Check API health
	@echo "🏥 Checking API health..."
	@curl -f http://localhost:8000/health > /dev/null 2>&1 && echo "✅ API is healthy!" || echo "❌ API health check failed!"

# Show container logs
logs: ## Show container logs
	@echo "📋 Showing container logs..."
	@docker-compose logs -f 2>/dev/null || docker logs -f vector-db-api 2>/dev/null || echo "No containers running"

# Show container status
status: ## Show container status
	@echo "📊 Container status:"
	@docker ps -a --filter "name=vector-db-api" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "📊 Docker Compose services:"
	@docker-compose ps 2>/dev/null || echo "No compose services running"

# Clean up Docker resources
clean: ## Clean up Docker resources
	@echo "🧹 Cleaning up Docker resources..."
	@docker system prune -f
	@docker volume prune -f
	@echo "✅ Cleanup completed!"

# Full setup (build + run)
setup: build run ## Build and run production container
	@echo "🎉 Setup complete! API is running at http://localhost:8000"

# Development setup (build + run dev)
dev-setup: build-dev dev ## Build and run development container
	@echo "🎉 Development setup complete! API is running at http://localhost:8000 with hot reloading"

# Quick test (build + test)
quick-test: build-dev test ## Build development image and run tests
	@echo "🎉 Quick test complete!"

# Restart production container
restart: stop run ## Restart production container

# Restart development container  
dev-restart: dev-stop dev ## Restart development container

