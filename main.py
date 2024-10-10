import cv2
from deepface import DeepFace

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break  # Exit loop if there's an issue capturing frames

    # Analyze the emotions in the current frame
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Extract emotions and their percentages
        emotions = result[0]['emotion']
        emotion = result[0]['dominant_emotion']
    except Exception as e:
        emotions = {"Error": 100}  # Display error if analysis fails

    # Define the text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White color
    thickness = 1
    position = (10, 50)  # Initial position for text

    # Display the emotion percentages
    cv2.putText(frame, f'Emotion: {emotion}', (position[0], position[1] - 20), font, font_scale, color, thickness, cv2.LINE_AA)
    for i, (emotion, percentage) in enumerate(emotions.items()):
        text = f'{emotion}: {percentage:.2f}%'
        cv2.putText(frame, text, (position[0], position[1] + i * 20), font, font_scale, color, thickness, cv2.LINE_AA)
    

    # Show the frame with detected emotions
    cv2.imshow('Video Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
