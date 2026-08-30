mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("hidden_dim", HIDDEN_DIM)
mlflow.log_param("epochs", EPOCHS)
mlflow.log_param("optimizer", "Adam")
mlflow.log_param("dataset", "MNIST")
mlflow.log_param("seed", SEED)

for epoch in range(EPOCHS):
    # training and validation happen here

    mlflow.log_metric(
        "train_loss",
        final_train_loss,
        step=epoch
    )
    mlflow.log_metric(
        "val_accuracy",
        final_val_accuracy,
        step=epoch
    )

mlflow.log_metric("best_val_accuracy", best_val_accuracy)
mlflow.log_metric("final_val_accuracy", final_val_accuracy)
mlflow.log_metric("final_train_loss", final_train_loss)
