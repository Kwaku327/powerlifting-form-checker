# AI-Powered Powerlifting App

An AI-powered application that analyzes powerlifting attempts using computer vision to assess compliance with IPF (International Powerlifting Federation) technical rules. The app uses MediaPipe for pose estimation and provides real-time feedback on lift execution.

## Features

- **Lift Analysis**: Supports squat, bench press, and deadlift analysis
- **Rule Checking**: Automated checking of key IPF technical rules
- **Visual Feedback**: Displays pose estimation and key frames
- **Detailed Reports**: Joint angles and rule infractions clearly explained
- **User-Friendly Interface**: Built with Streamlit for easy interaction

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/powerlifting_app.git
   cd powerlifting_app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   cd src
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the app:
   - Select your lift type from the sidebar
   - Upload a video of your lift
   - Wait for the analysis
   - Review the results and feedback

## Best Practices for Video Recording

For optimal results:
- Record from a side angle (90 degrees)
- Ensure good lighting
- Wear fitted clothing
- Keep the camera stable
- Include the complete lift from setup to lockout
- Film in landscape mode
- Maintain a clear view of the lifter

## Technical Details

### Components

- `src/types.py`: Data classes for lifts and analysis results
- `src/pose_estimation.py`: MediaPipe-based pose detection
- `src/rule_checker.py`: IPF rule compliance logic
- `src/utils.py`: Visualization and helper functions
- `src/app.py`: Streamlit web interface

### Key Technologies

- **MediaPipe**: For human pose estimation
- **OpenCV**: Video processing and visualization
- **Streamlit**: Web interface
- **NumPy**: Numerical computations
- **Pydantic**: Data validation

## Limitations

- Analysis is approximate and should not replace human judges
- Accuracy depends on video quality and camera angle
- Some technical rules (like foot movement) may be hard to detect
- For competition purposes, always refer to official IPF judges

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for pose estimation
- [IPF Technical Rules](https://www.powerlifting.sport) for lift standards
- [Streamlit](https://streamlit.io/) for the web framework 