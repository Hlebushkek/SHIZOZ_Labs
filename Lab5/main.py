from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection

model_path = "./models/yolo-tiny.h5"
input_path = "./input/test_image.jpeg"
video_input_path = "./input/test_video.mp4"
output_path = "./output/output_image.jpeg"
video_output_path = "./output/output_video.mp4"

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

detection = detector.detectObjectsFromImage(
    input_image=input_path,
    output_image_path=output_path
)

for item in detection:
    print(f"{item['name']} {item['percentage_probability']}")

################################

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

videoDetector = VideoObjectDetection()
videoDetector.setModelTypeAsTinyYOLOv3()
videoDetector.setModelPath(model_path)
videoDetector.loadModel()

videoDetection = videoDetector.detectObjectsFromVideo(
    input_file_path=video_input_path,
    output_file_path=video_output_path,
    per_second_function=forSeconds
)