import cv2
import numpy as np

from utils.prepare_data import loadSyncScore


def text_update(frameIdx, syncScoreList, frameToWriteOn):
    if frameIdx < len(syncScoreList):
        _syncScore = round(syncScoreList[frameIdx], 2)
        if _syncScore > 0.5:
            cv2.putText(frameToWriteOn, str(_syncScore), (nVideoWidth, video_height // 2), font, 1.5,
                        (255, 255, 255), 2,
                        cv2.LINE_4)
        else:
            cv2.putText(frameToWriteOn, str(_syncScore), (nVideoWidth, video_height // 2), font, 1.5,
                        (255, 255, 255), 2,
                        cv2.LINE_4)
    else:
        cv2.putText(frameToWriteOn, '0', (nVideoWidth, video_height // 2), font, 1.5,
                    (255, 255, 255), 2,
                    cv2.LINE_4)


def drawBackground(frameToWriteOn, syncScoreForFrame):
    mask = cv2.inRange(frameToWriteOn, lower_green_threshhold, upper_green_threshhold)
    frameToWriteOn[mask != 0] = [int(255 * syncScoreForFrame), 0, 0]


# def fillBackground(syncScore):
#     backgroundHeight = int(video_height * syncScore)


syncScore = loadSyncScore('..\\results\\data\\USER_STUDY_SMILE',
                          'SyncScore-USER_STUDY_SMILE-(16, 4, 3, 1)-25_03_2022_12_50_15')[1]['syncScore']

video1 = cv2.VideoCapture('..\\resources\\videos\\1_P1_3MIN.mp4')
video2 = cv2.VideoCapture('..\\resources\\videos\\1_P2_3MIN.mp4')

slowFactor = 5

frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
lower_green_threshhold = np.array([0, 160, 0])
upper_green_threshhold = np.array([70, 255, 70])
video_width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
nVideoWidth = int(video_width * 0.85) - int(video_width * 0.15)
font = cv2.FONT_HERSHEY_SIMPLEX
print(frames1)

out = cv2.VideoWriter('..\\results\\videos\\1_MERGED_H264.mp4', cv2.VideoWriter_fourcc(*"FMP4"), 30,
                      (nVideoWidth * 2, video_height))

for i in range(frames1):
    _, frame1 = video1.read()
    _, frame2 = video2.read()
    # drawBackground(frame1, syncScore[i])
    # drawBackground(frame2, syncScore[i])
    mergedFrame = np.hstack((frame1[0:video_height, int(video_width * 0.15):int(video_width * 0.85)],
                             frame2[0:video_height, int(video_width * 0.15):int(video_width * 0.85)]))
    # text_update(i, syncScore, mergedFrame)

    out.write(mergedFrame)

    # if syncScore[i] > 0.2:
    #     for j in range(slowFactor - 1):
    #         out.write(mergedFrame)

    if i % 500 == 0:
        print(f'{int(i / frames1 * 100)}%')

video1.release()
video2.release()
out.release()
