#!/usr/bin/env python3


# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import sys
from datetime import datetime

if len(sys.argv) != 2:
    print("Usage: python monitor_evaluation.py <version_number>")
    print("Example: python monitor_evaluation.py 2")
    sys.exit(1)

version = int(sys.argv[1])
eval_id = {EVAL_ID}
timeout = {TIMEOUT}

# Look for evaluation file in main evaluations directory
# When run from claude_workspaces/eval_X/, evaluations are at ../../evaluations/
eval_file = f'../../evaluations/evaluation_{eval_id}.json'
script_start_time = time.time()

# Log that monitor was called
monitor_log = "monitor_calls.log"
with open(monitor_log, 'a') as log:
    log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Monitor called for version {version}\n")
    log.flush()

print(f"Monitoring for evaluation_{eval_id}.json version v{version} (timeout: {timeout}s from version creation)...")
print(f"Looking for file at: {eval_file}")
print(f"Current working directory: {os.getcwd()}")
print("\n📌 REMINDER: If code runs without crashing for the timeout period, it's a SUCCESS!")
print("📌 Create report_success.md for running code, report_failed.md only for crashes/errors on final attempt.")

# Log initial status
with open(monitor_log, 'a') as log:
    log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Looking for file: {eval_file}\n")
    log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] CWD: {os.getcwd()}\n")

# First, we need to find when the version was created
version_created_time = None
version_key = f'v{version}'

while True:
    # Check if we should timeout waiting for file or version
    elapsed_waiting = time.time() - script_start_time
    if elapsed_waiting > 600:  # 10 minutes max to wait
        print(f"\n✅ SUCCESS! After {elapsed_waiting:.1f}s, either:")
        print("- The evaluation file hasn't appeared, OR")
        print("- The version hasn't been added to the file")
        print("\nThis usually means your code is running successfully in the evaluation system!")
        print("\n⚠️ IMPORTANT:")
        print("1. DO NOT CREATE ANY NEW VERSIONS! The current version is likely running successfully.")
        print("2. The evaluation system is busy or has file locking delays.")
        print("3. Create report_success.md (NOT report_failed.md) to document your successful fix.")
        print("4. Running code = SUCCESS, even if it takes time to start!")
        with open(monitor_log, 'a') as log:
            log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Result: TIMEOUT (SUCCESS) - file or version not found after {elapsed_waiting:.1f}s\n")
        sys.exit(0)  # Success - likely running without errors

    if os.path.exists(eval_file):
        try:
            # Force filesystem sync to ensure we see latest changes
            os.sync()
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Get version creation time if not already found
            if version_created_time is None:
                if version_key in data.get('versions', {}):
                    version_created_time = data['versions'][version_key].get('created_timestamp')
                    if version_created_time:
                        print(f"\nVersion {version} was created at {datetime.fromtimestamp(version_created_time).isoformat()}")
                        elapsed_since_creation = time.time() - version_created_time
                        print(f"Time elapsed since version creation: {elapsed_since_creation:.1f}s")
                else:
                    # Version not created yet - wait
                    with open(monitor_log, 'a') as log:
                        log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Version v{version} not found yet in file\n")
                    time.sleep(1)
                    continue
            
            # ALWAYS CHECK FOR RESULTS FIRST before checking timeout
            if version_key in data.get('versions', {}):
                version_data = data['versions'][version_key]
                if version_data.get('evaluation_result', {}).get('status') != 'pending':
                    # Version has been evaluated - process result immediately
                    result = version_data['evaluation_result']
                    score = result.get('score', 'n.a.')
                    
                    # Log what we found
                    with open(monitor_log, 'a') as log:
                        log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Version v{version} detected, score: {score}\n")
                    
                    # Check if evaluation failed (score is n.a.)
                    if score == 'n.a.':
                        # Print the evaluation details for Claude to see
                        print("\n❌ Evaluation failed! Details:")
                        print(f"Error: {result.get('error', 'Unknown error')}")
                        print(f"Logs: {result.get('logs', 'No logs')[:50000]}...")  # First 50000 chars
                        print("\n📝 Next steps:")
                        print("- Try to fix the error and create a new version")
                        print("- Only create report_failed.md if this is your FINAL attempt after giving up")
                        with open(monitor_log, 'a') as log:
                            log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Result: FAILED (score=n.a.)\n")
                            log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Error details: {result.get('error', 'No details')}\n")
                        sys.exit(2)  # Failed
                    else:
                        print(f"\nSuccess! Score: {score}")
                        with open(monitor_log, 'a') as log:
                            log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Result: SUCCESS (score={score})\n")
                        sys.exit(0)  # Success
            
            # Only check timeout AFTER checking for results
            if version_created_time:
                elapsed = time.time() - version_created_time
                if elapsed > timeout:
                    print(f"\n✅ SUCCESS! The submission has been running for {elapsed:.1f}s (exceeded monitor timeout of {timeout}s).")
                    print("This means your fix worked - the code is running without crashing!")
                    print("\n⚠️ IMPORTANT:")
                    print("1. DO NOT CREATE ANY NEW VERSIONS! The current version is running successfully.")
                    print("2. The evaluation is just taking longer than expected to complete.")
                    print("3. Create report_success.md (NOT report_failed.md) to document your successful fix.")
                    print("4. Running code = SUCCESS, even if it's slow!")
                    with open(monitor_log, 'a') as log:
                        log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Result: TIMEOUT (SUCCESS) - submission running for {elapsed:.1f}s\n")
                    sys.exit(0)  # Success - code is running without errors
                    
        except Exception as e:
            print(f"\nError reading evaluation file: {e}")
            with open(monitor_log, 'a') as log:
                log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] ERROR reading file: {e}\n")
    else:
        # File doesn't exist yet - log periodically
        if int(elapsed_waiting) % 10 == 0:
            print(f"Still waiting for file... ({int(elapsed_waiting)}s elapsed)")
            with open(monitor_log, 'a') as log:
                log.write(f"[{datetime.now().isoformat()}] [Eval {eval_id}] Still waiting, file not found yet ({int(elapsed_waiting)}s)\n")
    
    time.sleep(1)