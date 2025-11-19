# PendantProp Web Server

Web interface for controlling pendant drop surface tension measurements.

## Quick Start

1. **Generate placeholder images** (first time only):
```bash
python scripts/create_placeholders.py
```

2. **Start the server**:
```bash
python -m pendantprop.server.app
```

3. **Access the interface**:
Open your browser to `http://localhost:5000`

## Workflow

### 1. Initialize Protocol
- Click "Initialize Protocol"
- Enter experiment tag (e.g., `exp_001`)
- Upload deck layout CSV file
- Click "Initialize"

### 2. Measure Wells
- Click "Measure Wells"
- Upload sample information CSV file
- Click "Start Measurements"

### 3. Monitor Progress
- **Left panel**: Live pendant drop camera feed
- **Right panel**: Toggle between dynamic ST plot and results plot
- **Status bar**: Shows current protocol status and last action

## Features

- ✅ Live pendant drop camera stream
- ✅ Real-time plot updates (refreshes every 2 seconds)
- ✅ Tab-based plot switching (Dynamic ST vs Results)
- ✅ Background measurement execution
- ✅ Mock camera support for simulation mode

## File Upload Requirements

### Layout CSV
Should define the deck configuration (containers and their positions).

### Sample Info CSV
Must contain these columns:
- `well ID`: Well identifier (e.g., A1, B2, 10A1)
- `sample ID`: Sample name/identifier

## Configuration

Edit `config/settings.json`:
- Set `general_settings.simulate: true` for mock mode
- Configure camera, robot, and measurement parameters

## Directory Structure

```
src/pendantprop/server/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── index.html
│   ├── input_initialisation.html
│   ├── input_measure_wells.html
│   └── about.html
└── static/
    ├── styles.css         # CSS stylesheet
    ├── plots_cache/       # Auto-generated plots
    └── placehold_images/  # Placeholder images
```

## Notes

- The server creates a **shared camera instance** to avoid conflicts
- Plots are saved to `src/pendantprop/server/static/plots_cache/`
- Measurements run in background threads
- Server runs on port 5000 by default
