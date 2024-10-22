class APIRef(object):
    def __init__(self, win, source):
        self._device = cv2.VideoCapture(source)
        if isinstance(source, str):
            self.paused = True
        else:
            self.paused = False

        self.win = win
        cv2.namedWindow(self.win, 1)
        self.rect_selector = RectSelector(self.win, self.on_rect)
        self._bounding_box = None

        self.grabbed_frame = None
        self.zoomed = None
        
        self.write_video=False
        
        self.annotation = AnnotationLight("/home/poul/test.txt")
        self.a_iter = iter(self.annotation)
        self.annotation = None
        # self.a_iter = None
        
        self.prev_annotation = []
        self.ious = []
        self.scale_error = {
            "sx": [],
            "sy": []
        }
        self.shift_error = {
            "dx": [],
            "dy": []
        }

        self._tracker = MedianFlowTracker()

    def on_rect(self, rect):
        print("hi")
        self._bounding_box = rect
    
    def get_change(self, bb1: BoundingBox, bb2: BoundingBox):
        shift = bb1.center - bb2.center
        print("Shift is: ", shift)
        sx = (bb1.width) / (bb2.width)
        sy = (bb1.height) / (bb2.height)
        print("Scale change is: ", sx, sy)
        return shift, (sx, sy)

    def run(self):
        prev, curr = None, None
        ret, self.grabbed_frame = self._device.read()
        if self.a_iter:
            try: annot = next(self.a_iter)
            except StopIteration:
                print("Not able to extract annotation")
        else:
            annot = None
        if not ret:
            raise IOError('can\'t read frame')
        
        prev_bb = None
        prev_annot = []
        bb = []
        
        while True:
            if not self.rect_selector.dragging and not self.paused:
                ret, self.grabbed_frame = self._device.read()
                try: 
                    prev_annot = annot
                    annot = next(self.a_iter)
                    annot_box = BoundingBox.generate_from_list(annot)
                    prev_annot_box = BoundingBox.generate_from_list(prev_annot)
                    if self._bounding_box is None:
                        self.on_rect(prev_annot)
                except StopIteration: 
                    if not ret:
                        break

            frame = self.grabbed_frame.copy()
            prev, curr = curr, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None and self._bounding_box is not None and not self.paused:
                prev_bb = bb
                bb = self._tracker.update(self._bounding_box, prev, curr)
                if bb is not None:
                    self._bounding_box = bb
                    if annot:
                        self.ious.append(iou(bb, annot_box))
                    if prev_bb and bb is not None:
                        print("===============")
                        bb_sh, bb_s = self.get_change(bb, prev_bb)
                    if prev_annot and annot is not None:
                        print("In fact: ")
                        an_sh, an_s = self.get_change(annot_box, prev_annot_box)
                    if prev_bb and bb is not None and \
                        prev_annot and annot is not None:
                        sh_err = an_sh - bb_sh
                        sx, sy = an_s[0] / bb_s[0], an_s[1] / bb_s[1]
                        self.scale_error["sx"].append(sx)
                        self.scale_error["sy"].append(sy)
                        self.shift_error["dx"].append(sh_err.x)
                        self.shift_error["dy"].append(sh_err.y)
                        print("Shift error: ", sh_err)
                        print("Scale attitude: ", sx, sy)
                    
                    bb_list = map_int_list(list(bb))
                    cv2.rectangle(frame, bb_list[:2], bb_list[2:], (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, self._bounding_box[:2], self._bounding_box[2:], (0, 0, 255), 2)

            if annot:
                cv2.rectangle(frame, annot[:2], annot[2:], (255, 0, 0), 2)
            
            self.rect_selector.draw(frame)
            
                
            cv2.imshow(self.win, frame)
            if self.write_video:
                self.vid_orig_scale.write(frame)
            
            # if self._bounding_box and annot:
            #     box = list(annot)
            #     w, h = box[2] - box[0], box[3] - box[1]
            #     padding = [w//2, h//2]
            #     box[0] = max(0, box[0]-padding[0])
            #     box[1] = max(0, box[1]-padding[1])
            #     box[2] = max(0, box[2]+padding[0])
            #     box[3] = max(0, box[3]+padding[1])
            #     k = 5.
                
            #     cropped = np.array(self.grabbed_frame[box[1]:box[3], box[0]:box[2]])
            #     prev_cropped = np.array(prev[box[1]:box[3], box[0]:box[2]])
            #     prev_cropped = cv2.cvtColor(prev_cropped, cv2.COLOR_GRAY2BGR)
            #     cropped_size = cropped.shape[:2][::-1]
            #     desiered_size = [round(cropped_size[0]*k), 
            #                      round(cropped_size[1]*k)]
                
            #     cropped = cv2.resize(cropped, desiered_size, interpolation=cv2.INTER_NEAREST)
            #     prev_cropped = cv2.resize(prev_cropped, desiered_size, interpolation=cv2.INTER_NEAREST)
            #     img = np.hstack([prev_cropped, cropped])
            #     cropped = img
                
                
            #     if prev_bb is not None:
            #         prev_bb_transformed = [prev_bb[0]-box[0], prev_bb[1]-box[1], prev_bb[2]-box[0], prev_bb[3]-box[1]]
            #         prev_bb_transformed[0], prev_bb_transformed[1] = round(prev_bb_transformed[0]*k), round(prev_bb_transformed[1]*k)
            #         prev_bb_transformed[2], prev_bb_transformed[3] = round(prev_bb_transformed[2]*k), round(prev_bb_transformed[3]*k)
            #         cropped = cv2.rectangle(cropped, prev_bb_transformed[:2], prev_bb_transformed[2:], (0, 255, 255), 2)
                
            #     bb_list = map_int_list(list(bb))
            #     bb_transformed = [
            #         bb_list[0]-box[0], 
            #         bb_list[1]-box[1], 
            #         bb_list[2]-box[0], 
            #         bb_list[3]-box[1]
            #         ]
            #     bb_transformed[0], bb_transformed[1] = round(bb_transformed[0]*k), round(bb_transformed[1]*k)
            #     bb_transformed[2], bb_transformed[3] = round(bb_transformed[2]*k), round(bb_transformed[3]*k)
            #     bb_transformed[0] += desiered_size[0]
            #     bb_transformed[2] += desiered_size[0]
            #     cropped = cv2.rectangle(cropped, bb_transformed[:2], bb_transformed[2:], (0, 255, 255), 2)
                
            #     prev_annot_transformed = [prev_annot[0]-box[0], prev_annot[1]-box[1], prev_annot[2]-box[0], prev_annot[3]-box[1]]
            #     prev_annot_transformed[0], prev_annot_transformed[1] = round(prev_annot_transformed[0]*k), round(prev_annot_transformed[1]*k)
            #     prev_annot_transformed[2], prev_annot_transformed[3] = round(prev_annot_transformed[2]*k), round(prev_annot_transformed[3]*k)
            #     cropped = cv2.rectangle(cropped, prev_annot_transformed[:2], prev_annot_transformed[2:], (255, 255, 0), 2)
                
            #     annot_transformed = [annot[0]-box[0], annot[1]-box[1], annot[2]-box[0], annot[3]-box[1]]
            #     annot_transformed[0], annot_transformed[1] = round(annot_transformed[0]*k), round(annot_transformed[1]*k)
            #     annot_transformed[2], annot_transformed[3] = round(annot_transformed[2]*k), round(annot_transformed[3]*k)
            #     annot_transformed[0] += desiered_size[0]
            #     annot_transformed[2] += desiered_size[0]
            #     cropped = cv2.rectangle(cropped, annot_transformed[:2], annot_transformed[2:], (255, 255, 0), 2)
                
            #     for pt_init, pt_track in zip(self._tracker.ptsInit["good"], self._tracker.ptsTrack["good"]):
            #         center_init = [pt_init[0]-box[0], pt_init[1]-box[1]]
            #         center_init[0], center_init[1] = round(center_init[0]*k), round(center_init[1]*k)
                    
            #         center_track = [pt_track[0]-box[0], pt_track[1]-box[1]]
            #         center_track[0], center_track[1] = round(center_track[0]*k), round(center_track[1]*k)
            #         center_track[0]+=desiered_size[0]
                    
            #         cv2.circle(cropped, center_track, 2, (0, 255, 0), -1)
            #         cv2.line(cropped, center_init, center_track, (0, 255, 0), 1)
            #     for pt_init, pt_track in zip(self._tracker.ptsInit["bad"], self._tracker.ptsTrack["bad"]):
            #         center_init = [pt_init[0]-box[0], pt_init[1]-box[1]]
            #         center_init[0], center_init[1] = round(center_init[0]*k), round(center_init[1]*k)
                    
            #         center_track = [pt_track[0]-box[0], pt_track[1]-box[1]]
            #         center_track[0], center_track[1] = round(center_track[0]*k), round(center_track[1]*k)
            #         center_track[0]+=desiered_size[0]
                    
            #         cv2.circle(cropped, center_track, 2, (0, 0, 255), -1)
            #         cv2.line(cropped, center_init, center_track, (0, 255, 0), 1)
            #     cropped = cv2.resize(cropped, [640, 640], interpolation=cv2.INTER_NEAREST)
            #     cv2.imshow(self.win + "-cropped", cropped)

            if not self.paused and self.write_video:
                resized_crop = cropped = cv2.resize(cropped, self.writer_size, interpolation=cv2.INTER_NEAREST)
                self.vid.write(resized_crop)
            ch = cv2.waitKey(1)
            if ch == 27 or ch in (ord('q'), ord('Q')):
                break
            elif ch in (ord('p'), ord('P')):
                self.paused = not self.paused
