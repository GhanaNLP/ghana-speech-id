/* ghana-speech-id -- language identification over IPA phoneme strings.
 *
 * Identifies which of 41 Ghanaian and West African languages a phoneme string is in.
 * The input is the IPA that ghana-ipa-asr emits, so the full pipeline on device is
 *
 *     audio --[sherpa-onnx + ghana-speech-phoneme-asr]--> IPA units --[this]--> language
 *
 * No Python, no JSON parser, no STL in the ABI. Links against onnxruntime only; the
 * speech-to-IPA stage is a separate sherpa-onnx call that this library does not wrap,
 * so you can use the head on its own wherever the phonemes already exist.
 *
 * Accuracy rises steeply with how much speech the transcript came from. Measured out of
 * domain on real audio: three seconds scores about 0.51, whole clips averaging 9.7 seconds
 * about 0.78. Feed it five seconds or more, and check the audio is mostly speech before
 * transcribing -- silence contributes nothing and inflates the apparent duration.
 *
 * Thread safety: a GsidHead is safe for concurrent gsid_classify* calls. Creation and
 * destruction are not; do those from one thread.
 *
 * Licence: Apache-2.0 (code). The model weights follow the corpus licence, CC BY-NC 4.0.
 */
#ifndef GHANA_SPEECH_ID_H
#define GHANA_SPEECH_ID_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GSID_API
#  if defined(_WIN32) && defined(GSID_SHARED)
#    ifdef GSID_BUILDING
#      define GSID_API __declspec(dllexport)
#    else
#      define GSID_API __declspec(dllimport)
#    endif
#  else
#    define GSID_API
#  endif
#endif

typedef struct GsidHead GsidHead;

/* Paths to the artefacts written by scripts/export_onnx.py.
 * config_path may be NULL, in which case the n-gram range defaults to 1..5. */
typedef struct {
  const char *onnx_path;    /* head.onnx or head.fp16.onnx        */
  const char *ngrams_path;  /* ngrams.txt, one n-gram per line    */
  const char *labels_path;  /* labels.txt, one language per line  */
  const char *config_path;  /* head_config.txt, or NULL           */
  int num_threads;          /* onnxruntime intra-op threads; 0 => 1 */
} GsidConfig;

/* Fills cfg with NULLs and num_threads = 1. */
GSID_API void gsid_config_init(GsidConfig *cfg);

/* Returns NULL on failure and writes a message into err (if err != NULL). */
GSID_API GsidHead *gsid_create(const GsidConfig *cfg, char *err, size_t err_len);
GSID_API void gsid_destroy(GsidHead *head);

GSID_API int gsid_num_languages(const GsidHead *head);
/* Borrowed pointer, valid until gsid_destroy. NULL if index is out of range. */
GSID_API const char *gsid_language(const GsidHead *head, int index);

typedef struct {
  int index;         /* into the label list; -1 when undetermined */
  float confidence;  /* softmax probability of that label; 0 when undetermined */
  int num_matched;   /* n-grams found in the vocabulary; 0 means no decision was possible */
} GsidResult;

/* ipa: NUL-terminated UTF-8, units separated by ASCII spaces, exactly as
 * ghana-ipa-asr's Transcript.spaced() produces. Multi-codepoint units such as
 * k͡p, kʰ and t͡ʃ are single tokens and must not be split.
 *
 * A string whose n-grams are all out of vocabulary yields index = -1 with
 * num_matched = 0. Surface that to the user as "unknown" -- it means there was no
 * basis for a decision, not that some language scored poorly. */
GSID_API GsidResult gsid_classify(GsidHead *head, const char *ipa);

/* As above but also writes the full posterior. probs must have room for
 * gsid_num_languages floats. Returns the number written, or 0 if undetermined. */
GSID_API int gsid_classify_probs(GsidHead *head, const char *ipa, float *probs);

/* Version string of this library, e.g. "0.1.0". */
GSID_API const char *gsid_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* GHANA_SPEECH_ID_H */
