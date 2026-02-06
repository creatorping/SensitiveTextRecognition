"""
清理旧的checkpoint文件
只保留best_model.pt
"""
import os
import glob

def clean_old_checkpoints():
    """删除所有checkpoint_epoch_*.pt文件"""

    # 查找所有checkpoint文件
    checkpoint_files = glob.glob('checkpoint_epoch_*.pt')

    if not checkpoint_files:
        print("✓ 没有找到旧的checkpoint文件")
        return

    print(f"找到 {len(checkpoint_files)} 个旧checkpoint文件:")
    for f in checkpoint_files:
        print(f"  - {f}")

    # 确认删除
    response = input("\n是否删除这些文件? (y/n): ")

    if response.lower() == 'y':
        for f in checkpoint_files:
            try:
                os.remove(f)
                print(f"✓ 已删除: {f}")
            except Exception as e:
                print(f"✗ 删除失败 {f}: {e}")
        print(f"\n✓ 清理完成！删除了 {len(checkpoint_files)} 个文件")
    else:
        print("取消删除")

if __name__ == "__main__":
    print("="*60)
    print("清理旧Checkpoint文件")
    print("="*60)
    print("此脚本将删除所有 checkpoint_epoch_*.pt 文件")
    print("只保留 best_model.pt")
    print("="*60 + "\n")

    clean_old_checkpoints()
