import cv2  # OpenCV for video processing
import face_recognition  # Face recognition library
import os  # For file and directory handling

# Function: Load known faces and their encodings
def get_face_encodings(folder_path):
    """
    Loads face images from the given folder, calculates their encodings, 
    and extracts clean names from filenames.
    """
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    face_encodings = []
    face_names = []

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):  # Check for valid image extensions
            img_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(img_path)  # Load the image
            encodings = face_recognition.face_encodings(image)  # Get face encodings
            if encodings:  # Ensure the face is detected
                face_encodings.append(encodings[0])  # Store the first encoding
                clean_name = filename.split(".")[0].replace("_", " ").title()  # Clean the name
                face_names.append(clean_name)

    return face_encodings, face_names

# Path to known faces folder
known_faces_path = "./sample_faces"
known_face_encodings, known_face_names = get_face_encodings(known_faces_path)

# Start capturing video
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()  # Capture a frame from the webcam
    if not ret:  # Stop if the capture fails
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Faster HOG model
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale up face locations to match the original frame size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Compare face encoding with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

        # Default name is "Unknown"
        name = "Unknown"
        if True in matches:  # If there's a match
            match_index = matches.index(True)
            name = known_face_names[match_index]  # Retrieve the matched name

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Label the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the video feed
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()