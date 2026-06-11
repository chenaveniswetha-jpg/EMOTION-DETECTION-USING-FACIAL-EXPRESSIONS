from deepface import DeepFace
import cv2

# Enter your image path here
image_path = "myphoto.jpg.png"

try:
    # Analyze emotion
    result = DeepFace.analyze(
        img_path=image_path,
        actions=['emotion'],
        enforce_detection=False
    )

    emotion = result[0]['dominant_emotion']

    print("Detected Emotion:", emotion)

    # Display image
    img = cv2.imread(image_path)

    cv2.putText(
        img,
        f"Emotion: {emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Photo Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("Error:", e)