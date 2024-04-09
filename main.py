import pandas as pd

# Thay thế 'duong_dan_toi_file.csv' bằng đường dẫn thực tế của tệp CSV của bạn
duong_dan_file_csv = 'Training.csv'

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv(duong_dan_file_csv)

# Chọn cột gần cuối cùng (chẳng hạn, cột cuối cùng là -1, cột kế cuối là -2)
cot_gan_cuoi_cung = df.iloc[:, -2]
cot_gan_cuoi_cung = cot_gan_cuoi_cung.drop_duplicates()

# Phân loại dữ liệu (nếu cần)
# Ví dụ: df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')

# Nối lại các giá trị thành một chuỗi, sử dụng dấu phẩy làm phân tách
chuoi_ket_qua = ','.join(cot_gan_cuoi_cung.astype(str))

# In ra màn hình hoặc thực hiện các thao tác tiếp theo với chuỗi kết quả
print(chuoi_ket_qua)