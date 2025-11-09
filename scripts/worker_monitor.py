#!/usr/bin/env python3
"""
Worker Monitor Script for Ray Cluster GPU Health

Monitors GPU health using nvidia-smi and automatically disconnects from Ray cluster
if GPU failures persist for too many consecutive checks.

Usage:
    python worker_monitor.py                    # Monitor and disconnect from Ray on failures
    python worker_monitor.py --station          # Also stop the station when disconnecting
    
Or run as daemon:
    nohup python worker_monitor.py > worker_monitor.log 2>&1 &
    nohup python worker_monitor.py --station > worker_monitor.log 2>&1 &
"""

import subprocess
import time
import logging
import sys
import os
import argparse
from datetime import datetime


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('worker_monitor.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_initial_gpu_count():
    """
    Get initial GPU count using nvidia-smi.
    
    Returns:
        int: Number of GPUs detected, or None if nvidia-smi fails
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logging.error(f"nvidia-smi failed with return code {result.returncode}")
            logging.error(f"Error output: {result.stderr}")
            return None
        
        gpu_info = result.stdout.strip()
        if not gpu_info:
            logging.warning("No GPU information returned from nvidia-smi")
            return 0
        
        gpu_count = len(gpu_info.split('\n'))
        logging.info(f"Detected {gpu_count} GPUs using nvidia-smi")
        return gpu_count
        
    except subprocess.TimeoutExpired:
        logging.error("nvidia-smi command timed out during initial GPU detection")
        return None
    except FileNotFoundError:
        logging.error("nvidia-smi not found - no GPU drivers installed?")
        return None
    except Exception as e:
        logging.error(f"Unexpected error getting initial GPU count: {e}")
        return None


def check_gpu_health(expected_gpu_count):
    """
    Check GPU health using nvidia-smi.
    
    Args:
        expected_gpu_count: Expected number of GPUs from initialization
    
    Returns:
        bool: True if GPUs are healthy, False otherwise
    """
    try:
        # Run nvidia-smi to check GPU status
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logging.error(f"nvidia-smi failed with return code {result.returncode}")
            logging.error(f"Error output: {result.stderr}")
            return False
        
        # Parse output to check GPU availability
        gpu_info = result.stdout.strip()
        if not gpu_info:
            actual_gpu_count = 0
        else:
            actual_gpu_count = len(gpu_info.split('\n'))
        
        # Check if GPU count dropped from initial count
        if actual_gpu_count < expected_gpu_count:
            logging.error(f"GPU count dropped: nvidia-smi shows {actual_gpu_count}, expected {expected_gpu_count}")
            return False
        elif actual_gpu_count > expected_gpu_count:
            logging.warning(f"GPU count increased: nvidia-smi shows {actual_gpu_count}, expected {expected_gpu_count}")
        
        # Test basic CUDA functionality
        cuda_test = subprocess.run([
            'python3', '-c', 
            'import subprocess; subprocess.run(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"], check=True)'
        ], capture_output=True, timeout=15)
        
        if cuda_test.returncode != 0:
            logging.error("CUDA functionality test failed")
            return False
            
        logging.info(f"GPU health check passed - {actual_gpu_count} GPUs available and working")
        return True
        
    except subprocess.TimeoutExpired:
        logging.error("nvidia-smi command timed out")
        return False
    except FileNotFoundError:
        logging.error("nvidia-smi not found - no GPU drivers installed?")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during GPU health check: {e}")
        return False


def is_ray_worker_running():
    """
    Check if this node is connected to Ray cluster.
    
    Returns:
        bool: True if Ray is accessible, False otherwise
    """
    try:
        # Simply check if ray status command works
        result = subprocess.run(['ray', 'status'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logging.error("Ray status command timed out")
        return False
    except FileNotFoundError:
        logging.error("Ray command not found")
        return False
    except Exception as e:
        logging.error(f"Error checking Ray status: {e}")
        return False


def disconnect_from_ray(stop_station=False):
    """
    Disconnect this node from the Ray cluster.

    Args:
        stop_station: If True, also stop the station by running stop-production.sh

    Returns:
        bool: True if successfully disconnected, False otherwise
    """
    try:
        # Stop station FIRST if requested (before disconnecting from Ray)
        if stop_station:
            try:
                logging.warning("Stopping station by running stop-production.sh...")
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Go up one level to main station directory
                station_dir = os.path.dirname(script_dir)
                stop_script = os.path.join(station_dir, 'stop-production.sh')

                if os.path.exists(stop_script):
                    stop_result = subprocess.run([stop_script], capture_output=True, text=True, timeout=30)
                    if stop_result.returncode == 0:
                        logging.info("Successfully stopped station")
                    else:
                        logging.error(f"Failed to stop station: {stop_result.stderr}")
                else:
                    logging.error(f"stop-production.sh not found at {stop_script}")
            except Exception as e:
                logging.error(f"Error stopping station: {e}")

        logging.warning("Attempting to disconnect from Ray cluster due to persistent GPU failures...")

        # First try graceful shutdown
        result = subprocess.run(['ray', 'stop'], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            logging.info("Successfully disconnected from Ray cluster")
            return True
        else:
            logging.error(f"Ray stop failed: {result.stderr}")

            # Try more aggressive shutdown
            logging.warning("Trying more aggressive Ray process termination...")
            subprocess.run(['pkill', '-f', 'ray::'], timeout=10)
            subprocess.run(['pkill', '-f', 'raylet'], timeout=10)

            # Wait a bit and check if processes are gone
            time.sleep(5)
            if not is_ray_worker_running():
                logging.info("Ray processes terminated successfully")
                return True
            else:
                logging.error("Failed to terminate Ray processes")
                return False

    except subprocess.TimeoutExpired:
        logging.error("Ray stop command timed out")
        return False
    except Exception as e:
        logging.error(f"Error disconnecting from Ray: {e}")
        return False


def main():
    """Main monitoring loop."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Monitor GPU health and disconnect from Ray on failures')
    parser.add_argument('--station', action='store_true',
                        help='Also stop the station (run stop-production.sh) when disconnecting from Ray')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("Starting GPU health monitor for worker node")
    if args.station:
        logger.info("Station will be stopped if GPU failures persist")
    
    # Get initial GPU count using nvidia-smi
    initial_gpu_count = get_initial_gpu_count()
    if initial_gpu_count is None:
        logger.error("Could not detect initial GPU count - exiting")
        sys.exit(1)
    elif initial_gpu_count == 0:
        logger.warning("No GPUs detected - will monitor for CUDA functionality only")
    else:
        logger.info(f"Initial GPU count: {initial_gpu_count} GPUs")
    
    logger.info("Monitoring GPU health every 60 seconds...")
    logger.info("Will disconnect from Ray after 10 consecutive failures")
    
    consecutive_failures = 0
    max_consecutive_failures = 10
    check_interval = 60  # seconds
    
    try:
        while True:
            # Check if we're still a Ray worker
            if not is_ray_worker_running():
                logger.info("Ray worker not running - monitor will exit")
                break
            
            # Check GPU health
            if check_gpu_health(initial_gpu_count):
                # GPU is healthy - reset failure counter
                if consecutive_failures > 0:
                    logger.info(f"GPU health restored after {consecutive_failures} failures")
                consecutive_failures = 0
            else:
                # GPU failed
                consecutive_failures += 1
                logger.warning(f"GPU health check failed ({consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"GPU has failed {consecutive_failures} consecutive times")
                    logger.error("Disconnecting from Ray cluster to prevent failed job assignments")
                    
                    if disconnect_from_ray(stop_station=args.station):
                        logger.info("Successfully disconnected from Ray cluster")
                        if args.station:
                            logger.info("Station has been stopped")
                        logger.info("Monitor will exit - manual intervention required to rejoin cluster")
                        break
                    else:
                        logger.error("Failed to disconnect from Ray cluster")
                        logger.error("Manual intervention required")
                        # Continue monitoring in case the issue resolves
            
            # Wait before next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in monitoring loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()