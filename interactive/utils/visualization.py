"""
Visualization Utilities for Interactive Notebooks

This module provides standardized visualization functions for all interactive notebooks,
ensuring consistent and informative visualizations across different sections.

Author: AI Documentation Project
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationManager:
    """Central manager for all visualization utilities"""

    def __init__(self):
        self.color_palette = sns.color_palette("husl", 10)
        self.figure_size = (12, 8)
        self.dpi = 300

    def set_style(self, style='seaborn-v0_8'):
        """Set matplotlib style"""
        plt.style.use(style)

    def create_scatter_plot(self, x: pd.Series, y: pd.Series,
                          title: str = "Scatter Plot",
                          xlabel: str = "X", ylabel: str = "Y",
                          color_by: pd.Series = None,
                          size_by: pd.Series = None,
                          alpha: float = 0.7,
                          figsize: tuple = None,
                          interactive: bool = False,
                          save_path: str = None) -> None:
        """
        Create a scatter plot

        Args:
            x: X-axis data
            y: Y-axis data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color_by: Series to color points by
            size_by: Series to size points by
            alpha: Point transparency
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        if interactive:
            fig = px.scatter(
                pd.DataFrame({'x': x, 'y': y, 'color': color_by, 'size': size_by}),
                x='x', y='y',
                color='color' if color_by is not None else None,
                size='size' if size_by is not None else None,
                title=title,
                labels={'x': xlabel, 'y': ylabel},
                opacity=alpha
            )
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)

            if color_by is not None:
                scatter = ax.scatter(x, y, c=color_by, alpha=alpha, cmap='viridis')
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(x, y, alpha=alpha)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_line_plot(self, x: pd.Series, y: pd.Series,
                        title: str = "Line Plot",
                        xlabel: str = "X", ylabel: str = "Y",
                        multiple_lines: dict = None,
                        figsize: tuple = None,
                        interactive: bool = False,
                        save_path: str = None) -> None:
        """
        Create a line plot

        Args:
            x: X-axis data
            y: Y-axis data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            multiple_lines: Dictionary of multiple y-series
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        if interactive:
            if multiple_lines:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, name='Primary'))
                for name, y_data in multiple_lines.items():
                    fig.add_trace(go.Scatter(x=x, y=y_data, name=name))
            else:
                fig = px.line(pd.DataFrame({'x': x, 'y': y}), x='x', y='y', title=title)

            fig.update_layout(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                title=title
            )
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)

            ax.plot(x, y, linewidth=2, label='Primary')
            if multiple_lines:
                for name, y_data in multiple_lines.items():
                    ax.plot(x, y_data, linewidth=2, label=name, alpha=0.7)
                ax.legend()

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_histogram(self, data: pd.Series,
                        title: str = "Histogram",
                        xlabel: str = "Value",
                        bins: int = 30,
                        figsize: tuple = None,
                        interactive: bool = False,
                        save_path: str = None) -> None:
        """
        Create a histogram

        Args:
            data: Data to plot
            title: Plot title
            xlabel: X-axis label
            bins: Number of bins
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        if interactive:
            fig = px.histogram(pd.DataFrame({'value': data}), x='value', nbins=bins, title=title)
            fig.update_layout(xaxis_title=xlabel)
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
            ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(True, alpha=0.3)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_box_plot(self, data: pd.DataFrame,
                       title: str = "Box Plot",
                       figsize: tuple = None,
                       interactive: bool = False,
                       save_path: str = None) -> None:
        """
        Create a box plot

        Args:
            data: DataFrame to plot
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        if interactive:
            fig = px.box(data, title=title)
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
            data.boxplot(ax=ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_correlation_heatmap(self, data: pd.DataFrame,
                                  title: str = "Correlation Heatmap",
                                  figsize: tuple = None,
                                  interactive: bool = False,
                                  save_path: str = None) -> None:
        """
        Create a correlation heatmap

        Args:
            data: DataFrame to compute correlations
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        corr_matrix = data.corr()

        if interactive:
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title=title)
        else:
            fig, ax = plt.subplots(figsize=figsize or (12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(title, fontsize=14, fontweight='bold')

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series,
                               labels: list = None,
                               title: str = "Confusion Matrix",
                               figsize: tuple = None,
                               interactive: bool = False,
                               save_path: str = None) -> None:
        """
        Create a confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        if interactive:
            fig = px.imshow(cm, text_auto=True, aspect="auto", title=title,
                          labels=dict(x="Predicted", y="Actual", color="Count"))
        else:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_roc_curve(self, y_true: pd.Series, y_scores: pd.Series,
                        title: str = "ROC Curve",
                        figsize: tuple = None,
                        interactive: bool = False,
                        save_path: str = None) -> None:
        """
        Create a ROC curve

        Args:
            y_true: True labels
            y_scores: Predicted scores
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = np.trapz(tpr, fpr)

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {auc_score:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier', line=dict(dash='dash')))
            fig.update_layout(
                title=title,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True
            )
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_precision_recall_curve(self, y_true: pd.Series, y_scores: pd.Series,
                                   title: str = "Precision-Recall Curve",
                                   figsize: tuple = None,
                                   interactive: bool = False,
                                   save_path: str = None) -> None:
        """
        Create a precision-recall curve

        Args:
            y_true: True labels
            y_scores: Predicted scores
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = np.mean(precision)

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall, y=precision, name=f'PR Curve (Avg Precision = {avg_precision:.3f})'))
            fig.update_layout(
                title=title,
                xaxis_title='Recall',
                yaxis_title='Precision',
                showlegend=True
            )
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
            ax.plot(recall, precision, linewidth=2, label=f'PR Curve (Avg Precision = {avg_precision:.3f})')
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_dimensionality_reduction_plot(self, X: pd.DataFrame, y: pd.Series = None,
                                           method: str = 'pca',
                                           title: str = None,
                                           figsize: tuple = None,
                                           interactive: bool = False,
                                           save_path: str = None) -> None:
        """
        Create dimensionality reduction visualization

        Args:
            X: Feature data
            y: Target labels (optional)
            method: 'pca' or 'tsne'
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            title = title or f"PCA Visualization (Explained Variance: {sum(reducer.explained_variance_ratio_):.2%})"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            title = title or "t-SNE Visualization"
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        df_plot = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1],
            'Class': y.astype(str) if y is not None else 'All Data'
        })

        if interactive:
            fig = px.scatter(df_plot, x='Component 1', y='Component 2', color='Class',
                           title=title, opacity=0.7)
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)

            if y is not None:
                unique_classes = y.unique()
                for i, class_label in enumerate(unique_classes):
                    mask = y == class_label
                    ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                             label=f'Class {class_label}', alpha=0.7)
                ax.legend()
            else:
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            ax.grid(True, alpha=0.3)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_model_comparison_plot(self, metrics: pd.DataFrame,
                                   title: str = "Model Comparison",
                                   figsize: tuple = None,
                                   interactive: bool = False,
                                   save_path: str = None) -> None:
        """
        Create model comparison visualization

        Args:
            metrics: DataFrame with model metrics
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        if interactive:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Accuracy Comparison', 'Performance Metrics'),
                specs=[[{"type": "bar"}, {"type": "radar"}]]
            )

            # Bar chart for accuracy
            fig.add_trace(
                go.Bar(x=metrics.index, y=metrics['accuracy'], name='Accuracy'),
                row=1, col=1
            )

            # Radar chart for multiple metrics
            metric_cols = [col for col in metrics.columns if col != 'accuracy']
            for model in metrics.index:
                fig.add_trace(
                    go.Scatterpolar(
                        r=metrics.loc[model, metric_cols],
                        theta=metric_cols,
                        fill='toself',
                        name=model
                    ),
                    row=1, col=2
                )

            fig.update_layout(title=title, height=600)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (16, 6))

            # Bar chart for accuracy
            metrics['accuracy'].plot(kind='bar', ax=ax1, color=self.color_palette)
            ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)

            # Heatmap for multiple metrics
            if len(metrics.columns) > 1:
                sns.heatmap(metrics.T, annot=True, cmap='YlOrRd', ax=ax2)
                ax2.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_learning_curve_plot(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                                 test_scores: np.ndarray,
                                 title: str = "Learning Curve",
                                 figsize: tuple = None,
                                 interactive: bool = False,
                                 save_path: str = None) -> None:
        """
        Create learning curve visualization

        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            test_scores: Test scores
            title: Plot title
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, name='Training Score',
                                   line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, name='Validation Score',
                                   line=dict(color='red')))
            fig.update_layout(
                title=title,
                xaxis_title='Training Set Size',
                yaxis_title='Score',
                showlegend=True
            )
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)

            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.1, color='blue')

            ax.plot(train_sizes, test_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                           alpha=0.1, color='red')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Set Size', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

    def create_feature_importance_plot(self, feature_names: list, importances: np.ndarray,
                                     title: str = "Feature Importance",
                                     top_n: int = None,
                                     figsize: tuple = None,
                                     interactive: bool = False,
                                     save_path: str = None) -> None:
        """
        Create feature importance visualization

        Args:
            feature_names: List of feature names
            importances: Feature importance values
            title: Plot title
            top_n: Number of top features to show
            figsize: Figure size
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)

        if top_n:
            df_importance = df_importance.tail(top_n)

        if interactive:
            fig = px.bar(df_importance, x='importance', y='feature',
                        orientation='h', title=title)
        else:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
            ax.barh(df_importance['feature'], df_importance['importance'], color=self.color_palette[0])
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)

        if save_path:
            fig.write_image(save_path) if interactive else fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        fig.show() if interactive else plt.show()

# Global instance
viz_manager = VisualizationManager()

# Convenience functions
def scatter_plot(x, y, **kwargs):
    """Convenience function for scatter plot"""
    return viz_manager.create_scatter_plot(x, y, **kwargs)

def line_plot(x, y, **kwargs):
    """Convenience function for line plot"""
    return viz_manager.create_line_plot(x, y, **kwargs)

def histogram(data, **kwargs):
    """Convenience function for histogram"""
    return viz_manager.create_histogram(data, **kwargs)

def box_plot(data, **kwargs):
    """Convenience function for box plot"""
    return viz_manager.create_box_plot(data, **kwargs)

def correlation_heatmap(data, **kwargs):
    """Convenience function for correlation heatmap"""
    return viz_manager.create_correlation_heatmap(data, **kwargs)

def confusion_matrix_plot(y_true, y_pred, **kwargs):
    """Convenience function for confusion matrix"""
    return viz_manager.create_confusion_matrix(y_true, y_pred, **kwargs)

def roc_curve_plot(y_true, y_scores, **kwargs):
    """Convenience function for ROC curve"""
    return viz_manager.create_roc_curve(y_true, y_scores, **kwargs)

def precision_recall_curve_plot(y_true, y_scores, **kwargs):
    """Convenience function for precision-recall curve"""
    return viz_manager.create_precision_recall_curve(y_true, y_scores, **kwargs)

def dim_reduction_plot(X, y=None, **kwargs):
    """Convenience function for dimensionality reduction plot"""
    return viz_manager.create_dimensionality_reduction_plot(X, y, **kwargs)

def model_comparison_plot(metrics, **kwargs):
    """Convenience function for model comparison plot"""
    return viz_manager.create_model_comparison_plot(metrics, **kwargs)

def learning_curve_plot(train_sizes, train_scores, test_scores, **kwargs):
    """Convenience function for learning curve plot"""
    return viz_manager.create_learning_curve_plot(train_sizes, train_scores, test_scores, **kwargs)

def feature_importance_plot(feature_names, importances, **kwargs):
    """Convenience function for feature importance plot"""
    return viz_manager.create_feature_importance_plot(feature_names, importances, **kwargs)