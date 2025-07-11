# AI Powerlifting Form Checker

An AI-powered application that analyzes powerlifting videos to assess compliance with IPF (International Powerlifting Federation) rules. The application uses computer vision to analyze squat, bench press, and deadlift videos, providing feedback on form and rule compliance.

## Features

- Video upload and analysis
- Real-time pose estimation using OpenCV
- Automated rule checking for:
  - Squat depth and bar position
  - Bench press pause and bar path
  - Deadlift lockout and bar path
- User-friendly web interface built with Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kwaku327/powerlifting-form-checker.git
   cd powerlifting-form-checker
   ```

2. Create and activate a virtual environment:
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
   python -m streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL
3. Upload a powerlifting video
4. Select the lift type (squat, bench press, or deadlift)
5. Click "Analyze" to receive feedback on form and rule compliance

## Project Structure

```
powerlifting-form-checker/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── pose_estimation.py  # Pose estimation logic
│   ├── rule_checker.py     # IPF rule checking implementation
│   ├── models.py          # Data models
│   └── utils.py           # Utility functions
└── requirements.txt      # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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

- `src/models.py`: Data classes for lifts and analysis results
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

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for pose estimation
- [IPF Technical Rules](https://www.powerlifting.sport) for lift standards
- [Streamlit](https://streamlit.io/) for the web framework 