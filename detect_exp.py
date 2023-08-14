#etect emotions on a human face and print them as label


import cv2
from deepface import DeepFace
import os
import uuid
import time

def delete_temp_files():
    temp_folder = "temp"
    if os.path.exists(temp_folder):
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")

def detect_emotions_live():
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start the video capture (you may need to adjust the parameter 0 based on your camera setup)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Create a "temp" folder if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")

    try:
        count = 1
        while True:
            # Capture each frame from the video stream
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture a frame.")
                break

            # Convert the frame to grayscale (for face detection)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Loop through each detected face and predict the emotion
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                # Generate a unique filename for each detected face
                temp_face_path = f"temp/temp_face_{str(uuid.uuid4())}.jpg"

                # Save the detected face temporarily with a unique filename
                cv2.imwrite(temp_face_path, face)

                try:
                    # Predict the emotion using DeepFace
                    predictions = DeepFace.analyze(img_path=temp_face_path, actions=['emotion'], enforce_detection=False)

                    if isinstance(predictions, list):
                        emotion = predictions[0]['dominant_emotion']
                    else:
                        emotion = predictions['dominant_emotion']

                    print("Emotion Label:", emotion)

                except FileNotFoundError:
                    print("Error: Image file not found.")
                    emotion = "Unknown"
                except Exception as e:
                    print("Error during emotion detection:", e)
                    emotion = "Unknown"

                # Remove the temporary face image
                os.remove(temp_face_path)

                # Draw a rectangle around the face and put the predicted emotion label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Display the resulting frame
            cv2.imshow('Emotion Detection', frame)

            # Save the image every 10 seconds
            if count % 10 == 0:
                image_path = f"temp/frame_{count}.jpg"
                cv2.imwrite(image_path, frame)
                print(f"Image saved as {image_path}")

            # Delete the contents of the "temp" folder every 30 seconds
            if count % 10 == 0:
                delete_temp_files()

            count += 1

            # Press 'q' to exit the loop and stop the video capture
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Wait for 10 seconds before processing the next frame
            time.sleep(10)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")

    except Exception as e:
        print("An unexpected error occurred:", e)

    finally:
        # Release the video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions_live()
