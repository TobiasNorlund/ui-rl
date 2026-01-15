# Simple Data Entry task environment

Here we show how to set up a verifiable UI task environment, to be used for training Computer Use models using `ui-rl`.
In Simple Data Entry, the task is to copy-paste data from a spreadsheet, to be submitted into a form:  

<img width="1738" height="1082" alt="Screenshot From 2026-01-10 21-14-04" src="https://github.com/user-attachments/assets/9ee73171-9e17-4b2d-8496-680546d9d9f6" />

The environment consists of:

 - Code for managing the two browser windows via Playwright: `SimpleDataEntrySession`, see [session.py](session.py)
   - It uses a Playwright hook to evaluate whether submitted data matches any row in the spreadsheet
 - A FastAPI server that launches the `SimpleDataEntrySession` and exposes a simple HTTP api for an agent to execute computer actions such as clicks and receive screenshots, see [server.py](server.py) and [computer.py](computer.py)
 - A [start.sh](start.sh) script for launching everything: 
   1. Launch a virtual framebuffer (display) via `Xvfb` 
   2. Launch a VNC server (for debugging and testing) 
   3. Launch `mutter` - a window manager for GNOME's look and feel
   4. Launch the FastAPI server
 - A [Dockerfile](Dockerfile) for containerizing the app
 - A [Makefile](Makefile) to build and run the container


## Quickstart

```bash
# Docker build and run
make build run

# Take a screenshot
curl "localhost:8000?action_type=screenshot" > screen.png

# Right click at a coordinate
curl "localhost:8000?action_type=right_click&x=100&y=300" > screen.png

# Press CTRL+C
curl "localhost:8000?action_type=hotkey&key=ctrl%2Bc" > screen.png

# => See computer.py for a complete action list...

# Get progress
curl "localhost:8000/progress"
```

At the beginning `/progress` which will print:

```
{"submitted_row_indices":[0],"num_incorrect_submissions":0}
```

If you connect via VNC to `localhost:5900` and submit the form with data from one of the rows, and curl `/progress` again, it should now be updated. 
