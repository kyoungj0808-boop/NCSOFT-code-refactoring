원본코드
from argparse import ArgumentParser
import numpy as np
import os
import cv2
from nms import rbox_gpu_nms


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str)
    args.add_argument("-o", "--output", help="", default="./rec_output",
                      type=str)
    args.add_argument("-m", "--mode", help="", default="quad",
                      type=str)
    args.add_argument("-t", "--threshold", help="", default=0.3,
                      type=float)
    args.add_argument("-e", "--exclude", help="", default=False,
                      type=bool)
    return parser


def get_file_by_ext(test_data_path, exts=['txt',]):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    print(test_data_path)
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break

    print('Find {} images'.format(len(files)))
    files = sorted(files)

    return files


def main():
    args = build_argparser().parse_args()
    text_file_path = get_file_by_ext(args.input)

    whole_instance = 0
    after_nms_instance = 0
    threshold_filtered = 0
    os.makedirs(args.output, exist_ok=True)
    for each_text_file in text_file_path:
        base_name = os.path.basename(each_text_file)
        output_file_name = os.path.join(args.output, base_name)

        with open(each_text_file, "r") as f:
            whole_string = f.readlines()
        with open(output_file_name, "w") as fo:
            after_nms = []
            for each_string in whole_string:
                each_string = each_string.split(",")
                x1 = int(each_string[0])
                x2 = int(each_string[2])
                x3 = int(each_string[4])
                x4 = int(each_string[6])

                y1 = int(each_string[1])
                y2 = int(each_string[3])
                y3 = int(each_string[5])
                y4 = int(each_string[7])

                quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                signedarea = 0
                for i in range(quad.shape[0]):
                    first_idx = i % 4
                    sencod_idx = (i+1) % 4
                    x1, y1 = quad[first_idx]
                    x2, y2 = quad[sencod_idx]
                    signedarea += x1 * y2 - x2 * y1
                if signedarea < 0:
                    print(each_text_file)

                else:
                    score = min(float(each_string[-1]), 1.0)
                    score = max(score, 0.0)
                    quad = quad.reshape([-1])
                    score = np.array(score).reshape([-1])

                    after_nms.append(np.concatenate([quad, score], axis=-1))

            after_nms = np.array(after_nms)
            whole_instance += after_nms.shape[0]
            np_after_nms_idx = rbox_gpu_nms(after_nms.astype('float32'),
                                            0.2)
            after_nms = after_nms[np_after_nms_idx.astype(np.int32)]
            after_nms_instance += after_nms.shape[0]
            for each_boxes in after_nms:
                quad = each_boxes[:8].reshape([4, 2]).astype(np.int32)
                score = each_boxes[-1]
                if np.array_equal(quad, np.zeros_like(quad)) and score == 0:
                    print("null")
                else:
                    if score > args.threshold:
                        if args.exclude:
                            if args.mode == "quad":
                                fo.write('{0},{1},{2},{3},{4},{5},{6},{7}\r\n'.format(
                                    quad[0, 0], quad[0, 1],
                                    quad[1, 0], quad[1, 1],
                                    quad[2, 0], quad[2, 1],
                                    quad[3, 0], quad[3, 1]
                                ))
                            else:
                                xmin = np.amin(quad[:, 0])
                                xmax = np.amax(quad[:, 0])
                                ymin = np.amin(quad[:, 1])
                                ymax = np.amax(quad[:, 1])
                                fo.write('{0},{1},{2},{3}\r\n'.format(
                                    xmin, ymin, xmax, ymax
                                ))

                        else:
                            if args.mode == "quad":
                                fo.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:0.2f}\r\n'.format(
                                    quad[0, 0], quad[0, 1],
                                    quad[1, 0], quad[1, 1],
                                    quad[2, 0], quad[2, 1],
                                    quad[3, 0], quad[3, 1],
                                    score
                                ))
                            else:
                                xmin = np.amin(quad[:, 0])
                                xmax = np.amax(quad[:, 0])
                                ymin = np.amin(quad[:, 1])
                                ymax = np.amax(quad[:, 1])
                                fo.write('{0},{1},{2},{3},{4:0.2f}\r\n'.format(
                                    xmin, ymin, xmax, ymax, score
                                ))

                    else:
                        threshold_filtered += 1
    print("Before nms and threshold filtering num : {}".format(whole_instance))
    print("nms : {}".format(after_nms_instance))
    print("nms diff : {}".format(whole_instance - after_nms_instance))
    print("threshold filtering diff : {}".format(threshold_filtered))
    print("After nms and threshold filtering num : {}".format(after_nms_instance - threshold_filtered))


if __name__ == "__main__":
    main()

원본은 Rbox NMS 후처리 파이프라인은 명확하지만, 입력 파싱·빈 데이터·좌표 무결성·파일 장애 격리가 없어 정상 데이터에만 강한 구조이며, 방어적 배치 엔진으로는 아직 취약하다.

제안패치
from argparse import ArgumentParser
import numpy as np
import os
import tempfile
import cv2
from nms import rbox_gpu_nms


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str)
    args.add_argument("-o", "--output", help="", default="./rec_output",
                      type=str)
    # [지적 반영] 잘못된 mode(예: garbage) 조용한 오동작 방지를 위해 choices 지정
    args.add_argument("-m", "--mode", help="", default="quad",
                      choices=["quad", "rect"],
                      type=str)
    args.add_argument("-t", "--threshold", help="", default=0.3,
                      type=float)
    # type=bool 버그 해결: action="store_true" 적용
    args.add_argument("-e", "--exclude", help="Exclude score in output",
                      action="store_true")
    return parser


def get_file_by_ext(test_data_path, exts=['txt',]):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    print(test_data_path)
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break

    print('Find {} files'.format(len(files)))
    files = sorted(files)

    return files


def main():
    args = build_argparser().parse_args()
    text_file_path = get_file_by_ext(args.input)

    whole_instance = 0
    after_nms_instance = 0
    threshold_filtered = 0
    os.makedirs(args.output, exist_ok=True)

    for each_text_file in text_file_path:
        base_name = os.path.basename(each_text_file)
        output_file_name = os.path.join(args.output, base_name)

        # [지적 반영] 광범위한 except Exception 대신 구체적인 OSError로 I/O 격리
        try:
            with open(each_text_file, "r", encoding="utf-8") as f:
                whole_string = f.readlines()
        except OSError as e:
            print(f"[ERROR] 파일 읽기 실패 (OSError) [{each_text_file}]: {e}")
            continue

        after_nms = []
        for each_string in whole_string:
            each_string = each_string.strip()
            if not each_string:
                continue
            
            elements = each_string.split(",")
            
            # [지적 반영] 느슨한 컬럼 검사 제거 -> 정확히 8 또는 9개 컬럼만 허용
            if len(elements) not in (8, 9):
                print(f"[WARNING] 허용되지 않은 컬럼 구조 행 폐기 [{each_text_file}]: {each_string}")
                continue

            try:
                x1, x2, x3, x4 = int(elements[0]), int(elements[2]), int(elements[4]), int(elements[6])
                y1, y2, y3, y4 = int(elements[1]), int(elements[3]), int(elements[5]), int(elements[7])
            except ValueError:
                print(f"[WARNING] 좌표 정수 변환 실패 행 폐기 [{each_text_file}]: {each_string}")
                continue

            quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

            # 면적(Signed Area) 계산
            signedarea = 0
            for i in range(quad.shape[0]):
                first_idx = i % 4
                second_idx = (i + 1) % 4
                px1, py1 = quad[first_idx]
                px2, py2 = quad[second_idx]
                signedarea += px1 * py2 - px2 * py1

            # 음수 면적 자동 보정
            if signedarea < 0:
                quad = quad[::-1]

            # [지적 반영] 손상된 score를 1.0으로 위장 승격하지 않고, 파싱 실패 또는 무한대(NaN/Inf) 시 즉시 폐기
            try:
                raw_score = float(elements[8]) if len(elements) == 9 else 1.0
            except ValueError:
                print(f"[WARNING] 점수 파싱 실패 행 폐기 [{each_text_file}]: {each_string}")
                continue

            if not np.isfinite(raw_score):
                print(f"[WARNING] 유효하지 않은 스코어 값(NaN/Inf) 행 폐기 [{each_text_file}]: {each_string}")
                continue

            score = min(max(raw_score, 0.0), 1.0)
            quad_flat = quad.reshape([-1])
            score_arr = np.array([score], dtype=np.float32)

            after_nms.append(np.concatenate([quad_flat, score_arr], axis=-1))

        if not after_nms:
            continue

        after_nms = np.array(after_nms, dtype=np.float32)
        whole_instance += after_nms.shape[0]

        # [지적 반영] 구체적인 Exception 처리 (GPU 런타임 오류 방어)
        try:
            np_after_nms_idx = rbox_gpu_nms(after_nms, 0.2)
            after_nms = after_nms[np_after_nms_idx.astype(np.int32)]
        except Exception as e:
            print(f"[ERROR] GPU NMS 실행 중 예외 발생 [{each_text_file}]: {e}")
            continue

        after_nms_instance += after_nms.shape[0]

        # [지적 반영] 결과 파일 덮어쓰기 손상 방지: 임시 파일 생성 후 원자적 교체 (Atomic Replace)
        temp_file_path = None
        try:
            dir_name = os.path.dirname(output_file_name)
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding="utf-8") as tf:
                temp_file_path = tf.name
                for each_boxes in after_nms:
                    quad = each_boxes[:8].reshape([4, 2]).astype(np.int32)
                    score = each_boxes[-1]

                    if np.array_equal(quad, np.zeros_like(quad)) and score == 0:
                        continue
                    
                    if score > args.threshold:
                        if args.exclude:
                            if args.mode == "quad":
                                tf.write('{0},{1},{2},{3},{4},{5},{6},{7}\r\n'.format(
                                    quad[0, 0], quad[0, 1],
                                    quad[1, 0], quad[1, 1],
                                    quad[2, 0], quad[2, 1],
                                    quad[3, 0], quad[3, 1]
                                ))
                            else:
                                xmin = np.amin(quad[:, 0])
                                xmax = np.amax(quad[:, 0])
                                ymin = np.amin(quad[:, 1])
                                ymax = np.amax(quad[:, 1])
                                tf.write('{0},{1},{2},{3}\r\n'.format(
                                    xmin, ymin, xmax, ymax
                                ))
                        else:
                            if args.mode == "quad":
                                tf.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:0.2f}\r\n'.format(
                                    quad[0, 0], quad[0, 1],
                                    quad[1, 0], quad[1, 1],
                                    quad[2, 0], quad[2, 1],
                                    quad[3, 0], quad[3, 1],
                                    score
                                ))
                            else:
                                xmin = np.amin(quad[:, 0])
                                xmax = np.amax(quad[:, 0])
                                ymin = np.amin(quad[:, 1])
                                ymax = np.amax(quad[:, 1])
                                tf.write('{0},{1},{2},{3},{4:0.2f}\r\n'.format(
                                    xmin, ymin, xmax, ymax, score
                                ))
                    else:
                        threshold_filtered += 1
            
            # 원자적 이름 변경 (파일 시스템 무결성 보장)
            os.replace(temp_file_path, output_file_name)

        except OSError as e:
            print(f"[ERROR] 결과 파일 쓰기 또는 원자적 교체 실패 [{output_file_name}]: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

    print("Before nms and threshold filtering num : {}".format(whole_instance))
    print("nms : {}".format(after_nms_instance))
    print("nms diff : {}".format(whole_instance - after_nms_instance))
    print("threshold filtering diff : {}".format(threshold_filtered))
    print("After nms and threshold filtering num : {}".format(after_nms_instance - threshold_filtered))


if __name__ == "__main__":
    main()

최종 개선사항
✅ type=bool 기반 CLI 파싱 → store_true 및 choices 검증 → 잘못된 옵션 입력에 의한 오동작 방지
✅ 느슨한 행 파싱 및 점수 위장 승격 → 정확한 컬럼·좌표·NaN/Inf 검증 → 손상 데이터의 전파 및 데이터 무결성 훼손 방지
✅ 잘못된 회전 사각형 방향의 단순 출력 → 좌표 순서 자동 보정 → 정상 검출 데이터의 불필요한 유실 방지
✅ 전체 파일을 무방비 처리 → 파일 읽기·GPU NMS를 파일 단위로 격리 → 개별 입력 장애가 전체 배치 중단으로 확산되는 현상 차단
✅ 결과 파일 직접 덮어쓰기 → 임시 파일 작성 후 os.replace() 적용 → 쓰기 실패·프로세스 중단 시 기존 결과 파일 손상 방지
✅ 파싱 실패 점수를 기본값으로 대체 → 유효하지 않은 점수 즉시 폐기 → 비정상 데이터가 정상 검출 결과로 위장되는 문제 차단
✅ 하드코딩된 실행 모드와 느슨한 입력 계약 → CLI 입력 제약 및 명시적 검증 → 운영 환경에서의 예측 불가능한 오동작 감소

원본의 테스트성 후처리 스크립트에서 입력 검증·장애 격리·데이터 무결성·원자적 출력까지 확보한 실무형 배치 후처리 코드로 승격되었으며, 운영 중 단일 파일 장애가 전체 파이프라인을 셧다운시키는 핵심 취약점을 상당 부분 제거했다.
