import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import re


# Veriyi hazırlama ve eğitim/test setlerine ayırma
def prepare_data(file_path, target_column, max_encoder_length=6, max_prediction_length=1):
    # Veri setini yükle ve sütunları düzenle
    data = pd.read_excel(file_path)
    data.columns = [re.sub(r'[().]', '_', col) for col in data.columns]  # Sütun adlarını temizle

    # 'Date' ve 'Time' sütunlarını datetime formatına çevir
    data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
    data = data.drop(columns=['Date', 'Time'])  # 'Date' ve 'Time' sütunlarını kaldır

    # 'time_idx' ve 'group' sütunlarını ekle
    data['time_idx'] = data.index.astype(int)
    data['group'] = 0  # Tüm veriler için aynı grup ID'si olacak

    # Veriyi eğitim ve test setlerine ayırma (80% eğitim, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    # Bilinen değişkenler ve hedef değişken
    known_reals = data.columns.tolist()
    if target_column in known_reals:
        known_reals.remove(target_column)

    # Eğitim için TimeSeriesDataSet oluştur
    train_dataset = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target=target_column,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[target_column],
        add_relative_time_idx=True,
        add_target_scales=True,
    )

    # Test verisi için TimeSeriesDataSet oluştur
    test_dataset = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        target=target_column,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[target_column],
        add_relative_time_idx=True,
        add_target_scales=True,
    )

    return train_dataset, test_dataset, test_data


# Eğitim ve test veri yükleyicileri oluştur
def create_dataloaders(train_dataset, test_dataset, batch_size=64):
    # Eğitim için DataLoader
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=4)

    # Test için DataLoader
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

    return train_dataloader, test_dataloader


# Modeli test et ve sonuçları kaydet
def test_model_and_save_results(test_dataloader, model, trainer):
    # Modeli test et ve tahminleri al
    predictions = trainer.predict(model, dataloaders=test_dataloader)

    # Gerçek değerleri ve tahmin edilen değerleri çıkart
    true_values = []
    predicted_values = []

    for batch in test_dataloader:
        x, y = batch  # x giriş verisi, y gerçek hedef
        true_values.append(y)  # Gerçek değerler

    # Tahmin edilen değerleri işleyelim
    for pred in predictions:
        # Burada ortadaki kuantili (0.5 quantile) alıyoruz, tahmin edilen değerlerde bu genellikle ortadaki kuantildir
        # Eğer model sadece bir kuantil çıkarıyorsa, ilk bileşeni alabiliriz.
        if pred[0].size(1) > 1:
            predicted_values.append(pred[0][:, 1].squeeze())  # Orta kuantil
        else:
            predicted_values.append(pred[0].squeeze())  # Tek kuantil varsa direkt al

    # Tensor'ları birleştir
    true_values = torch.cat([y for y in true_values], dim=0).numpy()
    predicted_values = torch.cat([p for p in predicted_values], dim=0).numpy()

    # R² değerini hesapla
    r2 = r2_score(true_values, predicted_values)
    print(f"R² Skoru: {r2}")

    # Sonuçları bir CSV dosyasına kaydet
    results_df = pd.DataFrame({
        "Gerçek Değerler": true_values.flatten(),
        "Tahmin Edilen Değerler": predicted_values.flatten()
    })
    results_df.to_csv("test_results.csv", index=False)
    print("Test sonuçları test_results.csv dosyasına kaydedildi.")

    return r2


# PyTorch Lightning modülü
class TFTLightningModule(pl.LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.tft_model = tft_model

    def forward(self, x):
        # Eğer giriş tuple ise, ilk bileşeni alacağız
        if isinstance(x, tuple):
            x = x[0]  # Sözlük formatındaki veri

        return self.tft_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.tft_model(x)

        if isinstance(y_hat, tuple):  # Eğer y_hat bir tuple ise, ilk bileşeni al
            y_hat = y_hat[0]

        loss = self.tft_model.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.tft_model(x)

        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]

        loss = self.tft_model.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)


# Ana fonksiyon
def main():
    # Eğitim veri seti yolu
    file_path = '../data/AirQualityUCI.xlsx'
    target_column = "CO_GT_"

    # Eğitim ve test veri setlerini hazırla
    train_dataset, test_dataset, test_data = prepare_data(file_path, target_column)

    # DataLoader'ları oluştur
    train_dataloader, test_dataloader = create_dataloaders(train_dataset, test_dataset)

    # Modeli tanımla
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # Quantile loss
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Modeli PyTorch Lightning modülüne sarma
    lightning_model = TFTLightningModule(tft)

    # Erken durdurma ve model kontrol noktası oluşturma (train_loss izleniyor)
    early_stop_callback = EarlyStopping(monitor="train_loss", patience=3, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1)

    # Logger
    logger = TensorBoardLogger("lightning_logs", name="tft")

    # Trainer oluşturma
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
    )

    # Modeli eğitme
    trainer.fit(lightning_model, train_dataloader)

    # Test aşaması
    r2 = test_model_and_save_results(test_dataloader, lightning_model, trainer)

    # R² skorunu kontrol et
    if r2 > 0:
        print(f"R² skoru 0'ın üstünde: {r2}")
    else:
        print(f"R² skoru 0'ın altında: {r2}")


if __name__ == "__main__":
    main()
