import os
import shutil
from torch_geometric.datasets import (
    Planetoid,
    Coauthor,
    Amazon,
    Yelp,
    Reddit
)
from tqdm import tqdm
from collections import OrderedDict

DATASETS = OrderedDict([
    ('Cora', {
        'class': Planetoid,
        'root': '/tmp/Cora',
        'kwargs': {'name': 'Cora'},
        'approx_nodes': 2.7e3
    }),
    ('CiteSeer', {
        'class': Planetoid,
        'root': '/tmp/CiteSeer',
        'kwargs': {'name': 'CiteSeer'},
        'approx_nodes': 3.3e3
    }),
    ('PubMed', {
        'class': Planetoid,
        'root': '/tmp/PubMed',
        'kwargs': {'name': 'PubMed'},
        'approx_nodes': 19.7e3
    }),
    ('CoauthorCS', {
        'class': Coauthor,
        'root': '/tmp/CoauthorCS',
        'kwargs': {'name': 'CS'},
        'approx_nodes': 18.3e3
    }),
    ('CoauthorPhysics', {
        'class': Coauthor,
        'root': '/tmp/CoauthorPhysics',
        'kwargs': {'name': 'Physics'},
        'approx_nodes': 34.5e3
    }),
    ('AmazonComputers', {
        'class': Amazon,
        'root': '/tmp/AmazonComputers',
        'kwargs': {'name': 'Computers'},
        'approx_nodes': 13.8e3
    }),
    ('AmazonPhoto', {
        'class': Amazon,
        'root': '/tmp/AmazonPhoto',
        'kwargs': {'name': 'Photo'},
        'approx_nodes': 7.7e3
    }),
    ('AmazonProducts', {
        'class': Amazon,
        'root': '/tmp/AmazonProducts',
        'kwargs': {'name': 'Products'},
        'approx_nodes': 1.3e6
    }),
    ('Yelp', {
        'class': Yelp,
        'root': '/tmp/Yelp',
        'kwargs': {},
        'approx_nodes': 700e3
    }),
    ('Reddit', {
        'class': Reddit,
        'root': '/tmp/Reddit',
        'kwargs': {},
        'approx_nodes': 230e3
    })
])


def clear_corrupted_download(root_path):
    """Remove potentially corrupted files"""
    raw_dir = os.path.join(root_path, 'raw')
    if os.path.exists(raw_dir):
        print(f"🗑️  Cleaning {raw_dir}...")
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)


def save_dataset(data, dataset_name):
    """Save dataset in specified format"""
    os.makedirs(dataset_name, exist_ok=True)

    # Save nodes
    print(f"\n📝 Saving {dataset_name} nodes...")
    with open(f"{dataset_name}/nodes.txt", "w") as f:
        for i in tqdm(range(data.num_nodes), desc="Nodes", unit="nodes"):
            features = " ".join(map(str, data.x[i].tolist()))
            f.write(features + "\n")

    # Process edges
    print(f"🔗 Processing {dataset_name} edges...")
    edge_set = set()
    for u, v in tqdm(data.edge_index.t().tolist(), desc="Deduplicating", unit="edges"):
        edge_set.add(frozenset({u, v}))

    with open(f"{dataset_name}/edges.txt", "w") as f:
        for edge in tqdm(edge_set, desc="Writing", unit="edges"):
            f.write(f"{min(edge)} {max(edge)}\n")

    print(f"✅ {dataset_name} complete | Nodes: {data.num_nodes:,} | Edges: {len(edge_set):,}")


def download_dataset(dataset_info):
    """Download with retry logic"""
    for attempt in range(3):
        try:
            clear_corrupted_download(dataset_info['root'])
            return dataset_info['class'](root=dataset_info['root'], **dataset_info['kwargs'])[0]
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
            if attempt == 2:
                raise
    return None


def main():
    """Main processing flow"""
    print("🚀 Starting dataset processing ")
    print("=" * 60)

    for name, info in DATASETS.items():
        print(f"\n🌐 Dataset: {name} (≈{info['approx_nodes']:,.0f} nodes)")
        try:
            data = download_dataset(info)
            print(f"📥 Download successful | Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,}")
            save_dataset(data, name)
        except Exception as e:
            print(f"❌ Failed to process {name}: {str(e)}")
            continue

    print("\n🎉 All datasets processed!")


if __name__ == "__main__":
    main()