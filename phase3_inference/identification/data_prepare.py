import os
import pandas as pd

def extend_small_csvs(folder_path: str, n: int, target_rows: int, overwrite: bool = False):
    """
    폴더 내 CSV 파일들 중 사이즈가 작은 n개의 파일을 골라,
    각 파일의 마지막 행을 복제하여 target_rows 개수까지 확장하는 함수.
    """

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found.")
        return

    csv_with_size = [(f, os.path.getsize(os.path.join(folder_path, f))) for f in csv_files]
    sorted_csv = sorted(csv_with_size, key=lambda x: x[1])
    smallest_n = sorted_csv[:n]

    print(f"Selected {len(smallest_n)} files:")
    for f, s in smallest_n:
        print(f" - {f} ({s} bytes)")

    for filename, size in smallest_n:
        file_path = os.path.join(folder_path, filename)

        try:
            df = pd.read_csv(file_path, header=None)
        except Exception as e:
            print(f"{filename} read error: {e}")
            continue

        rows_now = len(df)
        if rows_now >= target_rows:
            print(f"{filename}: already {rows_now} rows >= {target_rows}. Skipping.")
            continue

        # ✅ 마지막 행 (행 1개든 여러 개든 안전하게 DataFrame 반환)
        last_row = df.tail(1)

        # ✅ 부족한 행 개수 계산
        missing_rows = target_rows - rows_now

        # ✅ 복제할 행 여러 개 만들기
        repeated_rows = pd.concat([last_row] * missing_rows, ignore_index=True)

        # ✅ 기존 데이터와 합치기
        df_extended = pd.concat([df, repeated_rows], ignore_index=True)

        # ✅ 저장
        save_path = file_path if overwrite else os.path.join(folder_path, f"extended_{filename}")
        df_extended.to_csv(save_path, index=False)


    print(" completed!")


extend_small_csvs(folder_path="./data/validate", n=1,target_rows=4,overwrite=False)

extend_small_csvs(
    folder_path="./data/cluster",
    n=5,                # 작은 파일 상위 3개
    target_rows=49,   # 1000행까지 확장
    overwrite=True     # 새 파일로 저장
)

