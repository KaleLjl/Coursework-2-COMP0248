import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import open3d as o3d
from utils import depth_to_pointcloud, collate_fn


class Evaluator:
    def __init__(self, model, test_dataset, config):
        """
        Args:
            model: TableSegmentationModel
            test_dataset: DepthTableDataset for testing
            config: Evaluation configuration dictionary
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            collate_fn=collate_fn
        )

        self.config = config
        self.class_names = ["Background", "Table"]

        if not os.path.exists(config["output_dir"]):
            os.makedirs(config["output_dir"])

        if not os.path.exists(os.path.join(config["output_dir"], "visualizations")):
            os.makedirs(os.path.join(config["output_dir"], "visualizations"))

    def evaluate(self):
        """Run evaluation on test set"""
        all_preds = []
        all_labels = []
        all_filenames = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move data to device
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                paths = batch['paths']

                # Get predictions
                logits = self.model(points)
                preds = torch.argmax(logits, dim=2)

                # Store results
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_filenames.extend(paths)

        # Concatenate results
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_labels = np.concatenate([l.flatten() for l in all_labels])

        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels)

        # Generate confusion matrix
        self._plot_confusion_matrix(all_preds, all_labels)

        # Print classification report
        print(classification_report(all_labels, all_preds, target_names=self.class_names))

        # Save results
        self._save_results(metrics)

        # Visualize some examples
        self._visualize_examples()

        return metrics

    def _calculate_metrics(self, preds, labels):
        """Calculate evaluation metrics"""
        # Calculate accuracy
        accuracy = (preds == labels).mean()

        # Calculate IoU for each class
        iou_scores = []
        for cls in range(2):  # Binary classification
            intersection = np.logical_and(preds == cls, labels == cls).sum()
            union = np.logical_or(preds == cls, labels == cls).sum()
            iou = intersection / (union + 1e-10)
            iou_scores.append(iou)

        # Calculate mean IoU
        mean_iou = np.mean(iou_scores)

        # Calculate precision and recall for table class
        true_positives = np.logical_and(preds == 1, labels == 1).sum()
        false_positives = np.logical_and(preds == 1, labels == 0).sum()
        false_negatives = np.logical_and(preds == 0, labels == 1).sum()

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        metrics = {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'background_iou': iou_scores[0],
            'table_iou': iou_scores[1],
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        return metrics

    def _plot_confusion_matrix(self, preds, labels):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        cm_path = os.path.join(self.config["output_dir"], "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

    def _save_results(self, metrics):
        """Save evaluation results to file"""
        results_path = os.path.join(self.config["output_dir"], "evaluation_results.txt")

        with open(results_path, 'w') as f:
            f.write("Table Segmentation Evaluation Results\n")
            f.write("===================================\n\n")

            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

    def _visualize_examples(self, num_examples=5):
        """Visualize and save some example predictions"""
        # Choose random samples
        indices = np.random.choice(len(self.test_loader.dataset), num_examples, replace=False)

        for idx in indices:
            sample = self.test_loader.dataset[idx]
            points = sample['points'].unsqueeze(0).to(self.device)
            labels = sample['labels']
            path = sample['path']

            # Get predictions
            with torch.no_grad():
                logits = self.model(points)
                preds = torch.argmax(logits.squeeze(0), dim=1).cpu()

            # Create point cloud visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).cpu().numpy())

            # Color based on predictions (red: table, blue: background)
            colors = np.zeros((len(preds), 3))
            colors[preds == 0] = [0, 0, 1]  # Background: Blue
            colors[preds == 1] = [1, 0, 0]  # Table: Red
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save point cloud
            filename = os.path.basename(path).split('.')[0]
            output_path = os.path.join(
                self.config["output_dir"], 
                "visualizations", 
                f"{filename}_pred.ply"
            )
            o3d.io.write_point_cloud(output_path, pcd)

            # Also save ground truth visualization
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(points.squeeze(0).cpu().numpy())

            colors_gt = np.zeros((len(labels), 3))
            colors_gt[labels == 0] = [0, 0, 1]  # Background: Blue
            colors_gt[labels == 1] = [1, 0, 0]  # Table: Red
            pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)

            output_path_gt = os.path.join(
                self.config["output_dir"], 
                "visualizations", 
                f"{filename}_gt.ply"
            )
            o3d.io.write_point_cloud(output_path_gt, pcd_gt)

    def inference(self, depth_image, intrinsic_matrix):
        """Run inference on a single depth image"""
        # Convert depth to point cloud
        points = depth_to_pointcloud(depth_image, intrinsic_matrix)

        # Convert to tensor
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            logits = self.model(points_tensor)
            preds = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()

        # Create colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color based on predictions (red: table, blue: background)
        colors = np.zeros((len(preds), 3))
        colors[preds == 0] = [0, 0, 1]  # Background: Blue
        colors[preds == 1] = [1, 0, 0]  # Table: Red
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd, preds
