import os
import shutil
import random
from pathlib import Path
from collections import Counter

# ================= CẤU HÌNH (BẠN CHỈNH SỬA Ở ĐÂY) =================
# Đường dẫn đến thư mục chứa dữ liệu gốc (đang chứa các folder class)
SOURCE_DIR = "Wonders of World"  

# Đường dẫn đến thư mục bạn muốn lưu dữ liệu đã chia
DEST_DIR = "Wonders of World Splitting"

# Tỷ lệ chia (Train, Valid, Test) - Tổng phải bằng 1.0
SPLIT_RATIO = (0.7, 0.2, 0.1)

# Seed để đảm bảo kết quả giống nhau mỗi lần chạy (Reproducibility)
RANDOM_SEED = 42

# Các đuôi file ảnh hợp lệ cần quét
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
# ==================================================================

def split_dataset(source, dest, ratio, seed=42):
    """
    Hàm phân chia dữ liệu từ source sang dest theo tỷ lệ ratio.
    """
    source_path = Path(source)
    dest_path = Path(dest)
    
    if not source_path.exists():
        print(f"❌ Lỗi: Thư mục nguồn '{source}' không tồn tại!")
        return

    # Thiết lập seed
    random.seed(seed)
    
    # Lấy danh sách các class (là các thư mục con trong source)
    classes = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not classes:
        print("⚠️ Cảnh báo: Không tìm thấy thư mục class nào trong thư mục nguồn.")
        return

    print(f"🔍 Đã tìm thấy {len(classes)} lớp (classes): {[c.name for c in classes]}")
    print(f"🚀 Bắt đầu phân chia dữ liệu theo tỷ lệ: Train={ratio[0]}, Valid={ratio[1]}, Test={ratio[2]}")
    print("-" * 50)

    total_images_moved = 0
    
    for class_dir in classes:
        class_name = class_dir.name
        
        # Lấy tất cả file ảnh trong folder class hiện tại
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS and f.is_file()]
        
        # Xáo trộn ngẫu nhiên danh sách ảnh
        random.shuffle(images)
        
        n_total = len(images)
        if n_total == 0:
            print(f"⚠️ Class '{class_name}' không có ảnh nào hợp lệ. Bỏ qua.")
            continue
            
        # Tính toán số lượng cho từng tập
        n_train = int(n_total * ratio[0])
        n_valid = int(n_total * ratio[1])
        # n_test lấy phần còn lại để đảm bảo không sót file nào do làm tròn
        n_test = n_total - n_train - n_valid 
        
        # Chia list ảnh
        train_imgs = images[:n_train]
        valid_imgs = images[n_train:n_train + n_valid]
        test_imgs = images[n_train + n_valid:]
        
        # Dictionary map giữa tên tập và list ảnh tương ứng
        splits = {
            'train': train_imgs,
            'valid': valid_imgs,
            'test': test_imgs
        }
        
        print(f"📂 Đang xử lý class '{class_name}': Tổng {n_total} ảnh -> Train: {len(train_imgs)}, Valid: {len(valid_imgs)}, Test: {len(test_imgs)}")

        # Thực hiện copy file
        for split_name, split_images in splits.items():
            # Tạo đường dẫn đích: dest / train / class_name
            save_dir = dest_path / split_name / class_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for img in split_images:
                # Dùng copy2 để giữ nguyên metadata của ảnh
                shutil.copy2(img, save_dir / img.name)
                
        total_images_moved += n_total

    print("-" * 50)
    print(f"✅ Hoàn tất! Đã phân chia tổng cộng {total_images_moved} ảnh.")
    print(f"📁 Dữ liệu mới được lưu tại: {dest_path.absolute()}")

if __name__ == "__main__":
    split_dataset(SOURCE_DIR, DEST_DIR, SPLIT_RATIO, RANDOM_SEED)