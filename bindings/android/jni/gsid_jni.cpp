// JNI glue for org.ghananlp.speechid.GhanaSpeechId.
//
// The handle is a jlong holding the GsidHead*. Java strings arrive as UTF-16 and JNI's
// GetStringUTFChars gives modified UTF-8, which differs from real UTF-8 for supplementary
// characters. None of the 176 IPA units are outside the BMP, so modified UTF-8 and UTF-8
// agree here -- but the Kotlin side still passes a ByteArray of real UTF-8 for the phoneme
// string, so the distinction can never bite if the inventory grows.

#include <jni.h>

#include <string>
#include <vector>

#include "ghana_speech_id.h"

namespace {

std::string jstr(JNIEnv *env, jstring s) {
  if (!s) return {};
  const char *c = env->GetStringUTFChars(s, nullptr);
  std::string out = c ? c : "";
  if (c) env->ReleaseStringUTFChars(s, c);
  return out;
}

GsidHead *as_head(jlong h) { return reinterpret_cast<GsidHead *>(h); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_ghananlp_speechid_GhanaSpeechId_nativeCreate(
    JNIEnv *env, jclass, jstring onnx, jstring ngrams, jstring labels, jstring config,
    jint threads, jobjectArray errOut) {
  const std::string s_onnx = jstr(env, onnx);
  const std::string s_ngrams = jstr(env, ngrams);
  const std::string s_labels = jstr(env, labels);
  const std::string s_config = jstr(env, config);

  GsidConfig cfg;
  gsid_config_init(&cfg);
  cfg.onnx_path = s_onnx.c_str();
  cfg.ngrams_path = s_ngrams.c_str();
  cfg.labels_path = s_labels.c_str();
  cfg.config_path = s_config.empty() ? nullptr : s_config.c_str();
  cfg.num_threads = threads;

  char err[512] = {0};
  GsidHead *h = gsid_create(&cfg, err, sizeof(err));
  if (!h && errOut && env->GetArrayLength(errOut) > 0) {
    env->SetObjectArrayElement(errOut, 0, env->NewStringUTF(err));
  }
  return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL
Java_org_ghananlp_speechid_GhanaSpeechId_nativeDestroy(JNIEnv *, jclass, jlong h) {
  gsid_destroy(as_head(h));
}

JNIEXPORT jint JNICALL
Java_org_ghananlp_speechid_GhanaSpeechId_nativeNumLanguages(JNIEnv *, jclass, jlong h) {
  return gsid_num_languages(as_head(h));
}

JNIEXPORT jstring JNICALL
Java_org_ghananlp_speechid_GhanaSpeechId_nativeLanguage(JNIEnv *env, jclass, jlong h,
                                                        jint index) {
  const char *s = gsid_language(as_head(h), index);
  return s ? env->NewStringUTF(s) : nullptr;
}

/* Returns {index, confidence, numMatched} as a float[3]; index and numMatched are exact
 * in float well past the 41 classes and any realistic n-gram count. */
JNIEXPORT jfloatArray JNICALL
Java_org_ghananlp_speechid_GhanaSpeechId_nativeClassify(JNIEnv *env, jclass, jlong h,
                                                        jbyteArray ipaUtf8) {
  std::string ipa;
  if (ipaUtf8) {
    const jsize n = env->GetArrayLength(ipaUtf8);
    ipa.resize(static_cast<size_t>(n));
    env->GetByteArrayRegion(ipaUtf8, 0, n, reinterpret_cast<jbyte *>(ipa.data()));
  }
  const GsidResult r = gsid_classify(as_head(h), ipa.c_str());
  const jfloat vals[3] = {static_cast<jfloat>(r.index), r.confidence,
                          static_cast<jfloat>(r.num_matched)};
  jfloatArray out = env->NewFloatArray(3);
  env->SetFloatArrayRegion(out, 0, 3, vals);
  return out;
}

/* Full posterior; returns null when nothing matched. */
JNIEXPORT jfloatArray JNICALL
Java_org_ghananlp_speechid_GhanaSpeechId_nativeClassifyProbs(JNIEnv *env, jclass, jlong h,
                                                             jbyteArray ipaUtf8) {
  std::string ipa;
  if (ipaUtf8) {
    const jsize n = env->GetArrayLength(ipaUtf8);
    ipa.resize(static_cast<size_t>(n));
    env->GetByteArrayRegion(ipaUtf8, 0, n, reinterpret_cast<jbyte *>(ipa.data()));
  }
  const int n = gsid_num_languages(as_head(h));
  std::vector<float> probs(static_cast<size_t>(n));
  const int got = gsid_classify_probs(as_head(h), ipa.c_str(), probs.data());
  if (got == 0) return nullptr;
  jfloatArray out = env->NewFloatArray(got);
  env->SetFloatArrayRegion(out, 0, got, probs.data());
  return out;
}

}  // extern "C"
