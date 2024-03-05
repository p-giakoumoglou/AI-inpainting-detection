# -*- coding: utf-8 -*-
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

def evaluate_model_performance(model, dataloader, csv_file_path):

    
    # Helper function to calculate Intersection over Union (IoU)
    def calculate_iou(true_mask, pred_mask):
        # Calculate Intersection and Union
        intersection = (true_mask & pred_mask).sum()
        union = (true_mask | pred_mask).sum()
        # Avoid division by zero
        iou = (intersection / union) if union != 0 else 0
        return iou.item()
    
    # Initialize dictionaries to store metrics for each image
    precision_per_image = {}
    recall_per_image = {}
    f1_score_per_image = {}
    auc_per_image = {}
    iou_per_image = {}
    max_prob_per_image = {}  # Stores the maximum probability as the score for each image
    actual_label = {}
    
    with torch.no_grad():
        for images, masks, paths in tqdm(dataloader, desc="Processing images"):
            images = images.cuda()
            masks = masks.cuda()
            
            #Problem. We shoudlnt reload the model but there is a problem 
            #with iddnet and for some reason it starts outputing NaN
            model.load('')
            model.eval()
            
            outputs = model(images)  # Assuming outputs are probabilities
            assert not torch.isnan(outputs).any(), "NaN values in outputs"
            
            predicted_masks = (outputs > 0.5).float()
            
            for i, path in enumerate(paths):
                # Calculate metrics
                pred_flat = predicted_masks[i].view(-1).cpu().numpy()
                true_flat = masks[i].view(-1).cpu().numpy()
                prob_flat = outputs[i].view(-1).cpu().numpy()  # Use the output probabilities 
                
                max_prob = np.max(prob_flat)
                max_prob_per_image[path] = max_prob
                
                if np.all(true_flat == 0):  # If the true image is real
                    actual_label[path] = 0
                    if max_prob < 0.5:  # No forgery detected
                        precision, recall, f1, auc, iou = (1,)*5
                    else:  # Forgery detected but the image is real
                        precision, recall, f1, auc, iou = (0,)*5
                else:
                    actual_label[path] = 1
                    
                    precision = precision_score(true_flat, pred_flat, zero_division=0)
                    recall = recall_score(true_flat, pred_flat, zero_division=0)
                    f1 = f1_score(true_flat, pred_flat, zero_division=0)
                    auc = roc_auc_score(true_flat, prob_flat)  # AUC requires the probabilities
                    iou = calculate_iou(torch.tensor(true_flat, dtype=torch.bool), torch.tensor(pred_flat, dtype=torch.bool))
                
                # Store metrics in dictionaries
                precision_per_image[path] = precision
                recall_per_image[path] = recall
                f1_score_per_image[path] = f1
                auc_per_image[path] = auc
                iou_per_image[path] = iou

    # Convert dictionaries to a DataFrame
    data = {
        'Path': list(precision_per_image.keys()),
        'F1': list(f1_score_per_image.values()),
        'Precision': list(precision_per_image.values()),
        'Recall': list(recall_per_image.values()),
        'IoU': list(iou_per_image.values()),
        'AUC': list(auc_per_image.values()),
        'Pred': list(max_prob_per_image.values()),
        'Actual': list(actual_label.values())
    }

    df = pd.DataFrame(data)

    # Optionally, save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    
    return df  # Return the DataFrame for immediate use

def calculate_pixel_image_metrics(df_path, output_path):
    # Load the DataFrame
    df = pd.read_csv(df_path)

    # Calculate the mean of the pixel-level scores
    mean_metrics = df[['F1', 'Precision', 'Recall', 'IoU', 'AUC']].mean()

    # Print average pixel-level scores
    print("Average Pixel-Level Scores:")
    print(mean_metrics)

    # Add a binary prediction column based on 'Pred' values
    df['Binary_Pred'] = (df['Pred'] > 0.5).astype(int)

    # Calculate image-level metrics
    image_level_metrics = {
        "Precision": precision_score(df['Actual'], df['Binary_Pred']),
        "Recall": recall_score(df['Actual'], df['Binary_Pred']),
        "F1-Score": f1_score(df['Actual'], df['Binary_Pred']),
        "Balanced Accuracy": balanced_accuracy_score(df['Actual'], df['Binary_Pred']),
        "AUC": roc_auc_score(df['Actual'], df['Pred'])
    }

    # Print image-level metrics
    print("Image-Level Metrics:")
    for metric_name, metric_value in image_level_metrics.items():
        print(f"{metric_name}: {metric_value}")

    # Prepare data for saving
    metrics_data = {
        "Metric Type": ["Average Pixel-Level"] * len(mean_metrics) + ["Image-Level"] * len(image_level_metrics),
        "Metric": list(mean_metrics.index) + list(image_level_metrics.keys()),
        "Value": list(mean_metrics.values) + list(image_level_metrics.values())
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Save the DataFrame to CSV
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

    return metrics_df


