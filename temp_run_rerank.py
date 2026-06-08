
import sys, os, uuid
sys.path.insert(0, r"D:\dataset_rag")
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-8d3c755454b643dfaf6e6a1405880ffa")

from app.query_process.agent.nodes.node_rerank import node_rerank

state = {
    "session_id": str(uuid.uuid4()),
    "is_stream": False,
    "query": "华为显示器怎么用",
    "embedding_chunks": [
        {"chunk_id": 1, "item_name": "华为显示器B3-211H", "chapter_name": "外观和接口介绍", "content": "显示器背面有HDMI、VGA、USB-C等多个接口，底部设有电源按键和OSD菜单按钮。", "embedding_score": 0.88},
        {"chunk_id": 2, "item_name": "华为显示器B3-211H", "chapter_name": "连接至计算机等设备", "content": "使用HDMI线连接显示器与计算机，支持4K@60Hz显示。USB-C接口支持视频传输与65W反向充电。", "embedding_score": 0.85},
        {"chunk_id": 3, "item_name": "华为显示器B3-211H", "chapter_name": "开关机及OSD菜单设置", "content": "短按底部摇杆键打开OSD菜单，可调节亮度、对比度、色温等参数。长按摇杆键3秒关机。", "embedding_score": 0.82},
        {"chunk_id": 4, "item_name": "华为显示器B3-211H", "chapter_name": "安装与拆卸显示器", "content": "将支架对准显示器背面卡扣位，听到咔嗒声表示安装到位。支持VESA壁挂安装。", "embedding_score": 0.80},
        {"chunk_id": 5, "item_name": "华为显示器B3-211H", "chapter_name": "常见问题", "content": "显示器无信号请检查信号线连接，画面闪烁请更新显卡驱动，色彩偏色可通过OSD菜单恢复出厂设置。", "embedding_score": 0.78}
    ]
}
state = node_rerank(state)
print("Rerank结果:")
for i, doc in enumerate(state.get("rerank_scored_docs", [])):
    print(f"  [{i+1}] score={doc['rerank_score']:.4f} | {doc['chapter_name']}")
print(f"Rerank完成，共 {len(state.get('rerank_scored_docs', []))} 条结果")
