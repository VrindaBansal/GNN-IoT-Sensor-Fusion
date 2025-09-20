#!/usr/bin/env python3
"""
Launch script for UrbanSense Web Demo
Combines the web visualization with real backend data
"""

import os
import sys
import subprocess
import threading
import time
import signal
import webbrowser
from pathlib import Path

def print_banner():
    """Print the UrbanSense banner"""
    print("=" * 80)
    print("🏙️  UrbanSense: Real-Time IoT Sensor Fusion Web Demo")
    print("🔬 Advancing Beyond Medical AI Research to Smart Cities")
    print("=" * 80)
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import dash
        import plotly
        import networkx
        import pandas
        import numpy
        import tensorflow
        print("✅ All Python dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please run: pip install -r requirements.txt")
        return False

def start_web_server():
    """Start the web visualization server"""
    web_demo_path = Path(__file__).parent / "web_demo"
    server_script = web_demo_path / "server.py"

    if not server_script.exists():
        print("❌ Web demo files not found")
        return None

    try:
        print("🌐 Starting web visualization server...")
        process = subprocess.Popen([
            sys.executable, str(server_script)
        ], cwd=str(web_demo_path))

        # Give server time to start
        time.sleep(2)
        return process

    except Exception as e:
        print(f"❌ Failed to start web server: {e}")
        return None

def start_backend_simulation():
    """Start the backend simulation with dashboard"""
    try:
        print("🔧 Starting backend simulation...")
        demo_script = Path(__file__).parent / "demo.py"

        if not demo_script.exists():
            print("❌ Demo script not found")
            return None

        # Start with dashboard disabled (we're using web demo instead)
        process = subprocess.Popen([
            sys.executable, str(demo_script),
            "--no-dashboard",
            "--sensors", "100",
            "--duration", "2.0"
        ])

        return process

    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def open_browser_tabs():
    """Open browser with both demos"""
    print("🚀 Opening browser tabs...")

    # Wait for servers to be ready
    time.sleep(3)

    try:
        # Open web demo
        webbrowser.open('http://localhost:8080')
        print("📱 Opened web demo at: http://localhost:8080")

        # Optional: Also open the Dash dashboard
        time.sleep(1)
        try:
            webbrowser.open('http://localhost:8050')
            print("📊 Opened Dash dashboard at: http://localhost:8050")
        except:
            pass  # Dash might not be running

    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("🌐 Please manually open: http://localhost:8080")

def main():
    """Main function to run the complete demo"""
    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    processes = []

    try:
        # Start web server
        web_process = start_web_server()
        if web_process:
            processes.append(web_process)
            print("✅ Web server started")

        # Start backend simulation
        backend_process = start_backend_simulation()
        if backend_process:
            processes.append(backend_process)
            print("✅ Backend simulation started")

        if not processes:
            print("❌ Failed to start any services")
            sys.exit(1)

        # Open browser
        browser_thread = threading.Thread(target=open_browser_tabs, daemon=True)
        browser_thread.start()

        print("\n🎯 Demo Features:")
        print("   • Real-time sensor network visualization")
        print("   • Live graph neural network processing")
        print("   • Smart city AI applications in action")
        print("   • Performance metrics and comparisons")
        print("\n🔍 What Makes This Better Than Research:")
        print("   • Real-time processing vs. batch analysis")
        print("   • Dynamic graphs vs. static relationships")
        print("   • Multiple applications vs. single use case")
        print("   • Production scale vs. research prototype")

        print("\n" + "=" * 60)
        print("🌐 WEB DEMO: http://localhost:8080")
        print("📊 DASH DASHBOARD: http://localhost:8050 (if available)")
        print("=" * 60)
        print("\nPress Ctrl+C to stop all services")

        # Wait for processes
        try:
            while True:
                time.sleep(1)
                # Check if any process has died
                for i, process in enumerate(processes[:]):
                    if process.poll() is not None:
                        print(f"⚠️  Process {i+1} has stopped")
                        processes.remove(process)

                if not processes:
                    print("🔚 All processes have stopped")
                    break

        except KeyboardInterrupt:
            print("\n⏹️  Stopping demo...")

    except Exception as e:
        print(f"❌ Error running demo: {e}")

    finally:
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass

        print("👋 UrbanSense demo stopped")

if __name__ == "__main__":
    main()