import cv2
from pathlib import Path
import aiofiles
import asyncio
import time 


async def get_markup_dict(annotation: Path):
    boxes = []
    async with aiofiles.open(annotation, "r") as annotation_file:
        await annotation_file.readline()  # Skip the header line
        async for line in annotation_file:
            boxes.append(line.strip().split(","))
    return {int(box[4]): list(map(int, box[:4])) for box in boxes}


async def process_vid(vid: cv2.VideoCapture, annotation, name="Video"):
    ret, frame = vid.read()
    if not ret:
        return
    new_annot_name = name.split(".")[0] + ".txt"
    with open(new_annot_name, "w") as annot:
        while ret:
            frame_number = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
            to_write = "None"
            if frame_number in annotation:
                pt1 = annotation[frame_number][:2]
                pt2 = annotation[frame_number][2:]
                pt2[0] += pt1[0]
                pt2[1] += pt1[1]
                to_write = f"{' '.join(map(str,pt1))}; {' '.join(map(str,pt2))}"
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255))
            annot.write(to_write+"\n")
            # cv2.imshow(name, frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            await asyncio.sleep(0)  # Yield control to allow async operations
            ret, frame = vid.read()

    # cv2.destroyWindow(name)


async def process_annotation(annotation: Path):
    annotation_boxes = await get_markup_dict(annotation)
    vid_folder = annotation.parents[1] / "Videos"
    vid_name = annotation.stem
    name = vid_name + ".webm"
    
    cap = cv2.VideoCapture(str(vid_folder / name))
    if not cap.isOpened():
        print(f"Cannot open video file: {vid_folder / name}")
        return

    try:
        await process_vid(cap, annotation_boxes, name)
    finally:
        cap.release()


async def main():
    annotation_folder = Path("/media/poul/8A1A05931A057E07/Job_data/Datasets/Thermal/Annotated_Vidos/Annotation/")
    annotation_files = annotation_folder.glob("*.csv")
    
    # Create tasks for each annotation file
    tasks = [process_annotation(annotation) for annotation in annotation_files]
    t = time.time()
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    print(time.time() - t)


if __name__ == "__main__":
    asyncio.run(main())
