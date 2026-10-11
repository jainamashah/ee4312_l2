# EE4312 L3 Task Division

Team of 4, so 4 new classes and about 400 images. Each member owns one class end to end (photos and labels) plus one stage of the pipeline.

| Deliverable | Who | Marks | Due |
| --- | --- | --- | --- |
| Annotated video `l3_team##.mp4` on Canvas, L3 | Team (C submits) | 26 | Sun 25 Oct 23:59. Our target is Sat 24 Oct |
| Written questions on Canvas, L3Q | Each member, individually | 24 | See Canvas |
| Q&A in lecture | Each member, individually | 50 | Fri 30 Oct |

## Roles

Photos and labels are split by class, so all four members work in parallel. Everything after the merge is split by pipeline stage. Replace A to D with names.

| Member | Class | Pipeline role | Notebook sections | Hands over |
| --- | --- | --- | --- | --- |
| A, dataset lead | 0 | Sets up the folder and `dataset.yaml`, merges all label sets, validates them | 2.1, 2.2, 2.4 (merge) | Clean `ee4312_l2` folder to B by Sat 17 Oct |
| B, training lead | 1 | Hosts the Drive folder, trains on a Colab T4, retrains after QA, archives runs | 2.6, 3, 4, 6 | `best.pt` to C and D by Mon 19 Oct |
| C, video lead | 2 | Records `sample.mp4`, generates the annotated video, submits it | 2.5, 5.2, 7 | Video on Canvas by Sat 24 Oct |
| D, QA lead | 3 | Labels the joint-shoot images, tests the model, checks the video against the rubric | 5.1, 7 | Pass/fail check before C submits |

B hosts the Drive folder because the notebook assumes `ee4312_l2` is at the top level of the Colab user's own My Drive.

## Shared rules (agree at kickoff)

1. **Classes.** Each member proposes one object. Check it against the [COCO class list](https://docs.ultralytics.com/datasets/detect/coco) (5 marks). Common traps are bottle, cup, mouse, keyboard, cell phone, book, scissors, clock, remote, laptop, backpack and teddy bear. Pick 3D objects whose silhouette changes when rotated, for example a stapler, tape dispenser, headphones or pencil case. Avoid four objects that look alike.
2. **Class order is fixed:** 0 = A's object, 1 = B's, 2 = C's, 3 = D's. Create the labels in exactly this order in makesense.ai. Use the same order in `dataset.yaml`. The local `dataset.yaml` currently lists only `class0` to `class2`, so add a `3:` entry.
3. **File names** start with your letter: `a_001.jpg`, `b_001.jpg`, and so on. Joint-shoot images use `j_001.jpg`. The notebook needs every file name to be unique.
4. **Image spec.** Make the long side 640 px. Take about 100 images per class. The object should fill 20% to 80% of the long side. Vary the angle and distance. Use at least one cluttered background as well as plain ones.
5. **Joint shoot** with all 4 objects (Thu 15 Oct):
   - About 40 images with 2 or more classes in frame. D labels every object in them.
   - About 20 background-only images of the same cluttered scenes, with no label file. These help the "minimal false identification" mark on empty frames.
   - C records `sample.mp4` in the same session.
6. **Export** from makesense.ai with *Actions > Export Annotations > YOLO format*. Send the zip and your images to A.
7. Use the same physical objects in the photos and the video.

## Checklists

### A, dataset lead (class 0)

- [ ] Create the folder structure (2.1) and `dataset.yaml` with all 4 class names (2.2)
- [ ] Shoot and label about 100 images of class 0
- [ ] Collect all images and label zips, then merge them into `images/` and `labels/`
- [ ] Run the check below. Fix any orphan labels and any class with too few boxes
- [ ] Spot-check about 10 boxes per class in makesense.ai
- [ ] Zip the folder and send it to B by Sat 17 Oct
- [ ] After QA, add the extra images D asks for, re-merge and re-send by Wed 21 Oct

```python
from pathlib import Path
from collections import Counter

root = Path("ee4312_l2/datasets/ee4312_l2")
imgs = {p.stem for p in (root / "images").iterdir()}
lbls = {p.stem for p in (root / "labels").glob("*.txt")}
print("labels with no image:", sorted(lbls - imgs))
print("images with no label (should only be background shots):", len(imgs - lbls))
counts = Counter(int(line.split()[0])
                 for f in (root / "labels").glob("*.txt")
                 for line in f.read_text().splitlines() if line.strip())
print("boxes per class:", dict(sorted(counts.items())))  # expect keys 0-3 only
```

### B, training lead (class 1)

- [ ] Shoot and label about 100 images of class 1
- [ ] Upload `ee4312_l2` to the top level of your own My Drive (2.6)
- [ ] In Colab, set the runtime to T4 GPU, then run 3.1 to 3.3 and 4.1 (`epochs=100`, `cache=False`). This takes about 1 hour
- [ ] Copy the runs to Drive right away (section 6), because Colab wipes `/content` when the session ends. First change `!cp /content/output.mp4` to `l3_team##.mp4`: the notebook copies a file that 5.2 never creates
- [ ] Send `best.pt` and the per-class mAP from `model.val()` to C and D by Mon 19 Oct
- [ ] Retrain on A's updated dataset on Wed 21 Oct and send the new `best.pt`

The train, val and test splits in `dataset.yaml` all point to the same images, so the validation mAP will look better than it really is. Judge the model on the sample video.

### C, video lead (class 2)

- [ ] Shoot and label about 100 images of class 2
- [ ] Record `sample.mp4` at the joint shoot (2.5). It must be 10 s or less, 640 px on the long side and about 50 MB or less. Suggested plan:
    - 0 to 2 s: empty cluttered background
    - 2 to 4 s: one object, rotating
    - 4 to 7 s: all 4 objects on the cluttered background, rotating
    - 7 to 9 s: all 4 objects on a plain background, from a different angle
- [ ] In 5.2, change `weights_path` from `"yolo26n.pt"` to the trained `best.pt`. The default is the COCO model, which labels COCO classes instead of ours
- [ ] Set `output_path` to `l3_team##.mp4` with our two-digit team number
- [ ] Make video v1 from the first model (Tue 20 Oct) and send it to D. Make the final version after the retrain (Thu 22 Oct)
- [ ] Before submitting, check the file name, that it is 10 s or less and 100 MB or less, and that it plays
- [ ] Submit on Canvas > Assignments > L3 on Sat 24 Oct. The upload takes about 10 min. Only one submission is needed per team

### D, QA lead (class 3)

- [ ] Shoot and label about 100 images of class 3
- [ ] At the joint shoot, take the background-only images and label every object in the `j_` images
- [ ] Run 5.1 on about 10 images per class, including new photos that are not in the training set. List missed or wrong classes
- [ ] Go through video v1 frame by frame. Log boxes on empty frames, wrong classes and missed classes. Send A a list of extra photos needed, for example "class 2 from above, 20 more" (Tue 20 Oct)
- [ ] Check the final video against the rubric below and give C the go-ahead (Fri 23 Oct)

| Rubric check (26 marks) | Marks |
| --- | --- |
| File is named `l3_team##.mp4` | 2 |
| At most 10 s | 2 |
| At most 100 MB | 2 |
| No class is in COCO | 5 |
| Minimal false IDs on frames with 0, 1 and several objects | 5 |
| Every class is identified from at least 2 perspectives | 5 |
| Every class is identified on at least 2 backgrounds, one of them cluttered | 5 |

## Timeline

| Dates | A | B | C | D | Gate |
| --- | --- | --- | --- | --- | --- |
| Sun 11 to Mon 12 Oct | Folder, yaml | | Video plan | | Classes, order and naming agreed |
| Tue 13 to Thu 15 Oct | Shoot and label class 0 | Shoot and label class 1 | Shoot and label class 2 | Shoot and label class 3 | Joint shoot and `sample.mp4`, Thu 15 Oct |
| Fri 16 to Sat 17 Oct | Merge and validate | | | Label `j_` images | Dataset sent to B, Sat 17 Oct |
| Sun 18 to Mon 19 Oct | | Upload, train run 1 | | | `best.pt` v1, Mon 19 Oct |
| Tue 20 to Wed 21 Oct | Add extra images | Retrain, Wed 21 Oct | Video v1 | QA images and video v1 | List of weak classes, Tue 20 Oct |
| Thu 22 to Fri 23 Oct | | Archive runs | Final video | Rubric check | Video passes, Fri 23 Oct |
| Sat 24 Oct | | | Submit | Confirm on Canvas | Submitted (hard deadline Sun 25 Oct 23:59) |
| Sun 25 to Thu 29 Oct | Everyone: L3Q answers and Q&A prep | | | | |
| Fri 30 Oct | Everyone: Q&A in lecture | | | | |

## Individual work (74 of 100 marks)

The written questions and the Q&A cannot be split. The GAs can ask anyone about any stage, so each member must be able to explain the whole pipeline, not just their own role.

- **L3Q (24 marks):** each member writes their own answers to notebook section 8. Do not share answers, because the plagiarism rule applies.
- **Q&A prep:** hold a 1-hour walkthrough on Mon 26 Oct where each lead explains their stage:
    - A: dataset and YOLO label format
    - B: transfer learning and training metrics
    - C: running the model on video
    - D: failure cases
- **Concepts to cover together, without writing shared answers:** the `Conv` arguments `[3, 16, 3, 2]`, parameter counts, spatial batch norm, bottleneck and residual connections.
