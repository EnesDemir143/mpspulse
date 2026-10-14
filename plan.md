# Apple Silicon PyTorch Profiling & Training ETA — Plan

## 1. Amaç ve kapsam

NVIDIA'nın [TensorRT-LLM performance skill'lerini](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills) **macOS / Apple Silicon / PyTorch MPS** için uyarlamak; gerçek benchmark ölçümlerinden eğitim süresi (ETA) üretmek ve gerektiğinde performans darboğazlarını analiz etmek.

- **Hedef:** M-serisi Mac'ler; ilk geliştirme/test cihazı **M2 Pro, 16 GB unified memory**.
- **Framework:** İlk sürümde genel **PyTorch MPS training loop** (CNN, ViT, segmentation ve özel `train_step`); mimariden bağımsız ölçüm, her projeyi otomatik tanıma garantisi yok.
- **Arayüz:** Bağımsız Python CLI + [Agent Skills](https://agentskills.io/specification) uyumlu skill'ler; OMP için slash command entegrasyonu.
- **Kapsam dışı (v1):** CUDA/NVIDIA, TensorRT-LLM serving, çoklu GPU, Slurm, özel Metal kernel yazımı, MLX ve bütün framework'lere otomatik destek.

## 2. Port edilecek NVIDIA skill'leri

| Kaynak | Yapılacak iş | Öncelik |
|---|---|---|
| [perf-workload-profiling](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-workload-profiling/SKILL.md) | Warmup, GPU/CPU timing, batch latency, throughput ve DataLoader ölçümlerini MPS'e port et | **v0.1** |
| [perf-analysis](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-analysis/SKILL.md) | Ölçümlere dayalı darboğaz sınıflandırmasını ve raporlamayı MPS için uyarlamak; NVIDIA specialist bağımlılıklarını çıkarmak | **v0.3** |
| [perf-torch-sync-free](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-torch-sync-free/SKILL.md) | Gereksiz host–device sync örüntülerini MPS'e göre tespit etmek; CUDA API'lerini kopyalamamak | **v0.4** |

**Port prensibi:** NVIDIA'nın *measure, don't guess*, warmup, düşük ölçüm overhead'i ve tekrarlanabilirlik ilkelerini koru. CUDA Events/NVTX/Nsight/CUDA Graph örneklerini birebir çevirmeye çalışma. Apache-2.0 lisans, telif/attribution ve değişiklik bildirimlerini koru.

## 3. macOS araçları ve görevleri

| Araç | Kullanım | Zorunluluk |
|---|---|---|
| [PyTorch MPS](https://docs.pytorch.org/docs/stable/mps.html) | `torch.mps.event.Event`, `synchronize()`, memory API'leri; **GPU timing** | Temel |
| Python `time.perf_counter()` | **End-to-end wall-clock**: data loading + CPU + GPU + optimizer | Temel |
| [`torch.mps.profiler`](https://docs.pytorch.org/docs/stable/mps.html) | OS Signposts / Metal trace; ileri profil | Opsiyonel |
| [macmon](https://github.com/vladkens/macmon) | JSON telemetry: RAM/swap, frekans, sıcaklık, GPU yükü; özel macOS API kullandığı için bağımlılık yapma | Opsiyonel |
| [Apple Metal Developer Tools](https://developer.apple.com/metal/tools/) | `metalperftrace` ile sistem izi; `gpucapture` → `gpudebug` ile ayrıntılı GPU trace | İleri seviye, desteklenen macOS'ta |
| Apple'ın [using-gpucapture](https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/using-gpucapture/SKILL.md) / [using-gpudebug](https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/using-gpudebug/SKILL.md) skill'leri | Derin trace iş akışı için referans | v0.4 |

> **Uyumluluk:** `gpucapture` / `gpudebug` macOS 27+ gerektirir. Önce kurulu araç/sürüm kontrolü yap; yoklarsa temel ETA bozulmamalı. Grafik odaklı trace metriklerinin ML compute workload'larına uygulanabilirliğini ayrıca doğrula.

## 4. Geliştirme aşamaları

### v0.1 — MPS workload profiling
- NVIDIA'nın timing kalıplarını port et: warmup → temsilî training step → GPU Events ve ayrı wall-clock ölçümü.
- DataLoader, forward/backward/optimizer, samples/sec, batch dağılımı ve MPS bellek ölçümlerini raporla.
- Senkronizasyon overhead'ini azalt; ölçüm sonuçlarını JSON olarak üret.
- **Bitti kriteri:** CNN + ViT + segmentation örneklerinde ölçümler tekrar üretilebiliyor; GPU ve end-to-end süreler karıştırılmıyor.

### v0.2 — Training ETA (özgün katkı)
- `steps/epoch`, epoch sayısı, gradient accumulation, validation, checkpoint ve startup maliyetlerini kullan.
- Modlar: `static` (düşük güven), `benchmark` (kısa ölçüm), `live` (gerçek adımlarla güncellenen ETA).
- Tek sayı yerine tahmin aralığı, ölçüm kaynağı, örnek sayısı ve uyarıları göster; geçmiş deneyleri sonraki tahminlere kalibrasyon için sakla.
- **Bitti kriteri:** Tahminler gerçekleşen toplam süreyle karşılaştırılıyor; yalnızca model/config'e bakarak uydurulan süreler **ölçüm** diye sunulmuyor.

### v0.3 — Performance analysis
- NVIDIA `perf-analysis` yaklaşımını MPS'e uyarla: compute / memory / data loading / host overhead / sync darboğazları.
- Hafif ölçüm yetmezse `torch.mps.profiler`, Instruments ve uygun sistemlerde `metalperftrace` kullan.
- **Bitti kriteri:** Her bulgu ham ölçüme dayanıyor; gerekçe ve öncelikli öneri üretiliyor.

### v0.4 — Sync ve deep GPU trace
- `perf-torch-sync-free` mantığıyla olası `.item()`, `.cpu()` vb. beklemeleri tanıla; otomatik kod değişikliği yapma.
- Araç varsa Apple `gpucapture` / `gpudebug` iş akışını isteğe bağlı bağla; gerekmiyorsa ağır trace toplama.
- **Bitti kriteri:** Deep profiling açıkça seçiliyor, normal ETA'yı profiler overhead'iyle kirletmiyor.

## 5. Mimari ve güvenlik kuralları

```text
skills/                 # SKILL.md: profiling, estimator, analysis, sync
commands/               # OMP: /train-estimate, /profile, /analyze-perf
src/                    # MPS timing, telemetry, ETA, reporting (CLI)
tests/                  # unit + macOS MPS integration + accuracy tests
references/             # NVIDIA port notları, Apple/PyTorch dokümanları
```

- **Skill = yönlendirme; Python CLI = gerçek ölçüm ve hesap.** LLM sayısal benchmark uyduramaz.
- Benchmark, model/optimizer/checkpoint'leri değiştirmemeli: mümkünse izole subprocess/deneme kopyası; tam eğitim veya kalıcı dosya değişikliği **açık izin** gerektirir.
- Çıktıda GPU event süresi ile pipeline wall-clock süresini **ayrı** tut. Bellek baskısı/swap, CPU fallback, değişken input şekli ve termal etkiler tahmin belirsizliğine girer.
- İlk doğrulama: M2 Pro üzerinde ResNet, ViT ve U-Net/SegFormer; farklı batch size ve resolution; ETA hata yüzdesi, tahmin aralığı kapsaması, profiling overhead ve ölçüm tekrarlanabilirliği.

## 6. İlk yapılacak işler

1. NVIDIA `perf-workload-profiling` ana dosya ve referanslarını sürüm/commit ile sabitle; lisans ve port notlarını ekle.
2. PyTorch MPS için küçük, **side-effect-safe** benchmark CLI prototipini geliştir.
3. M2 Pro'da üç model ailesinde gerçek wall-clock ölçümünü doğrula.
4. Ardından ETA motoru ve `/train-estimate` OMP command'ını bağla; ileri analiz skill'lerini sonraki sürümlere bırak.

**Başarı ölçütü:** macOS üzerinde, herhangi bir desteklenen PyTorch modelinin *temsilî eğitim adımları* için güvenilir profiling üretmek ve toplam eğitim süresini **ölçümlere dayalı, doğrulanabilir belirsizlikle** tahmin etmek.
