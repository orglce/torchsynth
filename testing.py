import IPython.display as ipd
import time
import torch
from typing import Optional
import torch
from torchsynth.module import (
    ADSR, LFO, VCA, AudioMixer, ModulationMixer,
    ControlRateUpsample, ControlRateVCA, SynthModule, MonophonicKeyboard,
    Noise, SineVCO, SquareSawVCO, TriangleVCO, Flanger
)
from torchsynth.synth import AbstractSynth
from torchsynth.config import SynthConfig

import midi_parser.test_midi as test_midi
import utils.audio_utils as u

class SimpleSynth(AbstractSynth):
    def __init__(self, synthconfig: Optional[SynthConfig] = None, device="cpu"):
        super().__init__(synthconfig=synthconfig, device=device)
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard),
                ("adsr", ADSR),
                # ("adsr_1", ADSR),
                ("upsample", ControlRateUpsample),
                ("vco", SineVCO),
                ("vcoTriangle", TriangleVCO),
                ("vco1", SquareSawVCO),
                ("flanger", Flanger),
                ("vca", VCA),
                # ("vca1", VCA),
                # (
                #     "mixer",
                #     AudioMixer,
                #     {
                #         "n_input": 2,
                #         "curves": [1.0, 1.0],
                #         "names": ["vco", "vco1"],
                #     },
                # ),
                
            ]
        )
        
        # self.set_hyperparameter(("keyboard", "duration", "curve"), 1.0)
       

    def output(self) -> torch.Tensor:
        envelope = self.adsr(self.midi_notes_length, self.midi_notes_length, self.midi_attacks, self.midi_decays)
        # envelope1 = self.adsr_1(note_on_duration)

        envelope = self.upsample(envelope)
        # time_plot(envelope.clone().detach().cpu().T, self.sample_rate.item())
        # envelope1 = self.upsample(envelope1)
        
        out = self.vco1(
            self.midi_notes, 
            self.midi_notes_start, 
            self.midi_notes_length,
            self.midi_attacks,
            self.midi_decays
        )
        
        # mixer rabimo da vse da skupi
        
        out = self.flanger(out)
        # out1 = self.vco1(midi_f0)
        # # Apply the amplitude envelope to the oscillator output
        out = self.vca(out, envelope)
        # out1 = self.vca(out1, envelope1)
        # time_plot(envelope.clone().detach().cpu().T, self.sample_rate.item())
        
        # out = self.mixer(out, out1)

        return out

device = "cuda"
sample_length = 7

notes, notes_start, notes_length, attacks, decays \
    = test_midi.preprocess_midi_files("assets/midi_files", device, num_of_samples=128, num_of_notes=15)

synthconfig = SynthConfig(batch_size=notes.shape[0], reproducible=False, buffer_size_seconds=sample_length, sample_rate=22050)

simplesynth = SimpleSynth(synthconfig, device)

start = time.time()
# new_frequency_param = torch.ones(synthconfig.batch_size, device="cuda")
# new_frequency_param[2] = torch.tensor([30])
# simplesynth.set_parameters({("flanger", "frequency"): new_frequency_param})

audio, parameters, is_train = simplesynth(None, notes, notes_start, notes_length, attacks, decays)
stft, db_magnitude, freqs = u.calc_stft(audio, device, sample_rate=simplesynth.sample_rate.item())
# u.plot_spectrogram(db_magnitude[0], freqs, int(simplesynth.sample_rate.item()), 512)

torch.cuda.synchronize()
print("Synthesis taken ", time.time() - start)
torch.cuda.empty_cache()
print(f"Created {audio.shape[0]} synthesizer sounds", f"that are each {audio.shape[1]} samples long")

params = simplesynth.get_parameters()

params_to_add = {
  "frequency": params[("flanger", "frequency")].data,
  "delay": params[("flanger", "delay")].data,
  "drive": params[("flanger", "drive")].data,
  "distortion_drive": params[("flanger", "distortion_drive")].data
}

u.write_wavs(audio, int(simplesynth.sample_rate.item()), "assets/audio_files", "file", params_to_add)

ipd.Audio(audio[0].cpu().numpy(), rate=int(simplesynth.sample_rate.item()), normalize=False, autoplay=True)