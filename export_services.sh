#!/bin/bash
# ============================================================
# export_services.sh - Expose VS Code and Jupyter via Cloudflare Tunnel
# ============================================================
# Usage:
#   ./export_services.sh                     # Export both VS Code and Jupyter
#   ./export_services.sh vscode              # Export only VS Code
#   ./export_services.sh jupyter             # Export only Jupyter
#   ./export_services.sh start <port>        # Export a specific port (e.g., start 8080)
#   ./export_services.sh status              # Show status of active tunnels
#   ./export_services.sh stop                # Stop all active tunnels
#   ./export_services.sh stop <port|service> # Stop a specific exported port or service
# ============================================================

set -euo pipefail

# Configuration
VS_CODE_PORT=30110
JUPYTER_PORT=30111
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$HOME/bin/cloudflared}"
PID_DIR="${HOME}/.local/cloudflared"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Ensure cloudflared is installed
check_cloudflared() {
    if ! command -v "$CLOUDFLARED_BIN" &>/dev/null; then
        log_error "cloudflared not found at $CLOUDFLARED_BIN"
        log_info "Installing cloudflared..."
        mkdir -p "$(dirname "$CLOUDFLARED_BIN")"
        curl -sSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
            -o "$CLOUDFLARED_BIN"
        chmod +x "$CLOUDFLARED_BIN"
        log_info "Installed: $( "$CLOUDFLARED_BIN" --version 2>&1 | head -1 )"
    fi
}

# Extract tunnel URL from cloudflared log
get_tunnel_url() {
    local log_file=$1
    grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$log_file" 2>/dev/null | head -1
}

# Start a tunnel for a given service
start_tunnel() {
    local service=$1
    local port=$2

    # Check if tunnel is already running for this service
    local pid_file="$PID_DIR/${service}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        log_warn "${service} tunnel is already running (PID: $(cat "$pid_file"))"
        return
    fi

    local log_file="$PID_DIR/${service}.log"
    log_info "Starting Cloudflare tunnel for ${service} (port ${port})..."

    mkdir -p "$PID_DIR"

    # Kill stale processes for this port
    pkill -f "cloudflared.*${port}" 2>/dev/null || true
    sleep 1

    # Start cloudflared
    "$CLOUDFLARED_BIN" tunnel --url "http://localhost:${port}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"

    # Wait for URL to appear (up to 120 seconds)
    local url=""
    local waited=0
    local max_wait=120
    for i in $(seq 1 $max_wait); do
        url=$(get_tunnel_url "$log_file") || true
        if [[ -n "$url" ]]; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
        if (( waited % 15 == 0 )); then
            log_info "  Waiting for tunnel... (${waited}s)"
        fi
    done

    if [[ -n "$url" ]]; then
        log_info "${GREEN}✅ ${service^} is live:${NC}"
        log_info "   URL: ${BLUE}${url}${NC}"
        return 0
    fi

    # Process is still running but URL not found yet
    if kill -0 "$pid" 2>/dev/null; then
        log_warn "Process still running (${waited}s), waiting more..."
        # Wait an additional 60 seconds
        for i in $(seq 1 60); do
            url=$(get_tunnel_url "$log_file") || true
            [[ -n "$url" ]] && break
            sleep 1
        done
        if [[ -n "$url" ]]; then
            log_info "${GREEN}✅ ${service^} is live:${NC}"
            log_info "   URL: ${BLUE}${url}${NC}"
            return 0
        fi
    fi

    # Retry once if process died
    log_warn "cloudflared exited. Retrying..."
    sleep 2
    > "$log_file"
    "$CLOUDFLARED_BIN" tunnel --url "http://localhost:${port}" > "$log_file" 2>&1 &
    pid=$!
    echo "$pid" > "$pid_file"
    url=""
    for i in $(seq 1 120); do
        url=$(get_tunnel_url "$log_file") || true
        [[ -n "$url" ]] && break
        sleep 1
    done
    if [[ -n "$url" ]]; then
        log_info "${GREEN}✅ ${service^} is live (retry):${NC}"
        log_info "   URL: ${BLUE}${url}${NC}"
        return 0
    fi

    log_error "Tunnel failed after retry. Check: $log_file"
    return 1
}

# Show status of all tunnels
show_status() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}Cloudflare Tunnel Status${NC}                                  ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}\n"

    local checked_services=()
    
    for service in vscode jupyter; do
        local pid_file="$PID_DIR/${service}.pid"
        local log_file="$PID_DIR/${service}.log"
        local port=$([[ "$service" == "vscode" ]] && echo "$VS_CODE_PORT" || echo "$JUPYTER_PORT")
        
        if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            local url=$(get_tunnel_url "$log_file")
            echo -e "  ${GREEN}✅ ${service^}${NC} (port ${port})"
            echo -e "     PID: $(cat "$pid_file")"
            [[ -n "$url" ]] && echo -e "     URL: ${BLUE}${url}${NC}"
        else
            echo -e "  ${RED}❌ ${service^}${NC} — not running"
        fi
        echo
        checked_services+=("$service")
    done
    
    if [[ -d "$PID_DIR" ]]; then
        for pid_file in "$PID_DIR"/*.pid; do
            [[ -e "$pid_file" ]] || continue
            local service=$(basename "$pid_file" .pid)
            if [[ " ${checked_services[*]} " == *" $service "* ]]; then continue; fi
            
            local log_file="$PID_DIR/${service}.log"
            local port="unknown"
            if [[ "$service" =~ ^port-([0-9]+)$ ]]; then port="${BASH_REMATCH[1]}"; fi
            
            if kill -0 "$(cat "$pid_file")" 2>/dev/null; then
                local url=$(get_tunnel_url "$log_file")
                echo -e "  ${GREEN}✅ ${service^}${NC} (port ${port})"
                echo -e "     PID: $(cat "$pid_file")"
                [[ -n "$url" ]] && echo -e "     URL: ${BLUE}${url}${NC}"
            else
                echo -e "  ${RED}❌ ${service^}${NC} (port ${port}) — not running"
            fi
            echo
        done
    fi
}

# Stop a specific tunnel
stop_tunnel() {
    local service=$1
    local pid_file="$PID_DIR/${service}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        kill "$(cat "$pid_file")"
        log_info "Stopped ${service} (PID: $(cat "$pid_file"))"
        rm -f "$pid_file" "$PID_DIR/${service}.log"
    else
        log_warn "${service} is not running."
    fi
}

# Stop all tunnels
stop_all() {
    log_info "Stopping all Cloudflare tunnels..."
    if [[ -d "$PID_DIR" ]]; then
        for pid_file in "$PID_DIR"/*.pid; do
            [[ -e "$pid_file" ]] || continue
            local service=$(basename "$pid_file" .pid)
            if kill -0 "$(cat "$pid_file")" 2>/dev/null; then
                kill "$(cat "$pid_file")"
                log_info "Stopped ${service} (PID: $(cat "$pid_file"))"
            fi
        done
        rm -rf "$PID_DIR"
    fi
    log_info "All tunnels stopped."
}

# Main
main() {
    check_cloudflared
    local action="${1:-all}"
    local arg2="${2:-}"

    case "$action" in
        vscode)
            start_tunnel "vscode" "$VS_CODE_PORT"
            ;;
        jupyter)
            start_tunnel "jupyter" "$JUPYTER_PORT"
            ;;
        start)
            if [[ -z "$arg2" ]]; then
                log_error "Please specify a port number (e.g., $0 start 8080)"
                exit 1
            fi
            start_tunnel "port-$arg2" "$arg2"
            ;;
        status)
            show_status
            ;;
        stop)
            if [[ -n "$arg2" ]]; then
                if [[ "$arg2" =~ ^[0-9]+$ ]]; then
                    stop_tunnel "port-$arg2"
                else
                    stop_tunnel "$arg2"
                fi
            else
                stop_all
            fi
            ;;
        all|*)
            start_tunnel "vscode" "$VS_CODE_PORT"
            echo
            start_tunnel "jupyter" "$JUPYTER_PORT"
            echo
            show_status
            ;;
    esac
}

main "$@"
