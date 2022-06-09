import os

from PIL import Image
from tqdm import tqdm

from RAANet import RAANet_
from utils.utils_metrics import compute_mIoU, show_results


if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 7
    # --------------------------------------------#
    #   区分的种类
    # --------------------------------------------#
    name_classes = ["null", "Impervious surfaces", "Building", "Low vegetation", "Tree ", "Car ", "background"]
    # -------------------------------------------------------#
    #   指向数据集所在的文件夹
    # -------------------------------------------------------#
    dataset_path = r'D:\work room\Liu\ISPRS 2D Semantic Labeling Contest\vaihingen\train_data'

    image_ids = open(os.path.join(dataset_path, "train_txt/train.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(dataset_path, "label/")
    miou_out_path = "new_da_aspp"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        net = RAANet_()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(dataset_path, "tif/" + image_id + ".tif")
            image = Image.open(image_path)
            image = net.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".tif"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
