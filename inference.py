
def gen_gambar(model,sport_predict,sports_chosen, path):
  with torch.no_grad():
    noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
    idx_encode = sports_chosen.index(sport_predict)
    labels = torch.tensor([idx_encode]).to(device)
    pic_predict = model(noise,labels)
    vutils.save_image(pic_predict, f"{path}/Test Image/Hasil predict gambar {sport_predict}.png", normalize = True)