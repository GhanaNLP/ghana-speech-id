// Implementation of the ghana-speech-id C API.
//
// The heavy lifting is in the ONNX graph, which carries the idf vector and the class
// weights and does the tf-idf arithmetic itself. This file only has to turn a phoneme
// string into (indices, counts) and call the session:
//
//   split on ASCII space -> emit 1..N-grams -> look each up -> count -> run
//
// The vocabulary is the one part worth being careful about. At 400k n-grams an
// unordered_map<string,int> costs tens of megabytes of small allocations, which is a lot
// to ask of a phone. Instead every n-gram lives back to back in one blob and lookup is a
// binary search over an offset table sorted by content: two allocations total, and the
// blob is exactly the bytes of ngrams.txt.

#include "ghana_speech_id.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <onnxruntime_cxx_api.h>

#define GSID_VERSION_STR "0.1.0"

namespace {

// ---------------------------------------------------------------- vocabulary

// n-grams stored contiguously; `sorted` indexes them in lexicographic order so lookup is
// a binary search. Index in the model is the *line number*, which is what `line_of` holds.
class Vocab {
 public:
  bool load(const std::string &path, std::string *err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { *err = "cannot open " + path; return false; }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) { *err = "cannot size " + path; return false; }
    in.seekg(0, std::ios::beg);

    blob_.resize(static_cast<size_t>(size));
    if (size > 0 && !in.read(blob_.data(), size)) { *err = "cannot read " + path; return false; }

    // one entry per line; keep offsets rather than copying substrings out
    offsets_.clear();
    size_t start = 0;
    for (size_t i = 0; i <= blob_.size(); ++i) {
      if (i == blob_.size() || blob_[i] == '\n') {
        size_t end = i;
        if (end > start && blob_[end - 1] == '\r') --end;   // tolerate CRLF
        if (i == blob_.size() && end == start) break;       // ignore a trailing newline
        offsets_.push_back({start, end - start});
        start = i + 1;
      }
    }

    sorted_.resize(offsets_.size());
    for (size_t i = 0; i < sorted_.size(); ++i) sorted_[i] = static_cast<int32_t>(i);
    std::sort(sorted_.begin(), sorted_.end(),
              [this](int32_t a, int32_t b) { return at(a) < at(b); });
    return true;
  }

  size_t size() const { return offsets_.size(); }

  std::string_view at(int32_t i) const {
    const auto &e = offsets_[static_cast<size_t>(i)];
    return std::string_view(blob_.data() + e.off, e.len);
  }

  // -1 when absent
  int32_t find(std::string_view s) const {
    size_t lo = 0, hi = sorted_.size();
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      const std::string_view v = at(sorted_[mid]);
      if (v < s) lo = mid + 1;
      else       hi = mid;
    }
    if (lo < sorted_.size() && at(sorted_[lo]) == s) return sorted_[lo];
    return -1;
  }

 private:
  struct Entry { size_t off; size_t len; };
  std::vector<char> blob_;
  std::vector<Entry> offsets_;
  std::vector<int32_t> sorted_;
};

bool read_lines(const std::string &path, std::vector<std::string> *out, std::string *err) {
  std::ifstream in(path);
  if (!in) { *err = "cannot open " + path; return false; }
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    out->push_back(line);
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------- the head

struct GsidHead {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "gsid"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Vocab vocab;
  std::vector<std::string> labels;
  int ngram_min = 1;
  int ngram_max = 5;

  std::string in0, in1, out0, out1;   // node names, owned
};

namespace {

// Splits on ASCII space only, so multi-byte units (k͡p, kʰ, t͡ʃ) survive intact.
void split_units(const char *ipa, std::vector<std::string_view> *out) {
  const char *p = ipa;
  while (*p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (!*p) break;
    const char *start = p;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') ++p;
    out->emplace_back(start, static_cast<size_t>(p - start));
  }
}

// Builds (indices, counts) for every in-vocabulary n-gram of order ngram_min..ngram_max.
// Counts are accumulated by sorting the hit list -- cheaper than a hash map for the few
// hundred n-grams a single utterance produces.
int featurise(const GsidHead *h, const char *ipa,
              std::vector<int64_t> *indices, std::vector<float> *counts) {
  std::vector<std::string_view> units;
  split_units(ipa, &units);
  if (units.empty()) return 0;

  std::vector<int32_t> hits;
  std::string gram;
  for (size_t i = 0; i < units.size(); ++i) {
    gram.clear();
    for (int n = 1; n <= h->ngram_max; ++n) {
      const size_t j = i + static_cast<size_t>(n) - 1;
      if (j >= units.size()) break;
      if (n > 1) gram += ' ';
      gram.append(units[j].data(), units[j].size());
      if (n < h->ngram_min) continue;
      const int32_t id = h->vocab.find(gram);
      if (id >= 0) hits.push_back(id);
    }
  }
  if (hits.empty()) return 0;

  std::sort(hits.begin(), hits.end());
  indices->clear(); counts->clear();
  for (size_t i = 0; i < hits.size();) {
    size_t j = i;
    while (j < hits.size() && hits[j] == hits[i]) ++j;
    indices->push_back(static_cast<int64_t>(hits[i]));
    counts->push_back(static_cast<float>(j - i));
    i = j;
  }
  return static_cast<int>(hits.size());
}

// Returns probs pointer into `owned` outputs, or nullptr when nothing matched.
const float *run(GsidHead *h, const char *ipa, int *n_matched,
                 std::vector<Ort::Value> *outputs) {
  std::vector<int64_t> indices;
  std::vector<float> counts;
  *n_matched = featurise(h, ipa, &indices, &counts);
  if (*n_matched == 0) return nullptr;

  const int64_t k = static_cast<int64_t>(indices.size());
  const int64_t shape[1] = {k};

  Ort::Value t_idx = Ort::Value::CreateTensor<int64_t>(
      h->mem, indices.data(), indices.size(), shape, 1);
  Ort::Value t_cnt = Ort::Value::CreateTensor<float>(
      h->mem, counts.data(), counts.size(), shape, 1);

  const char *in_names[2] = {h->in0.c_str(), h->in1.c_str()};
  const char *out_names[2] = {h->out0.c_str(), h->out1.c_str()};
  Ort::Value ins[2] = {std::move(t_idx), std::move(t_cnt)};

  *outputs = h->session->Run(Ort::RunOptions{nullptr}, in_names, ins, 2, out_names, 2);
  return (*outputs)[1].GetTensorData<float>();   // probs
}

}  // namespace

// ---------------------------------------------------------------- C API

extern "C" {

void gsid_config_init(GsidConfig *cfg) {
  if (!cfg) return;
  cfg->onnx_path = nullptr;
  cfg->ngrams_path = nullptr;
  cfg->labels_path = nullptr;
  cfg->config_path = nullptr;
  cfg->num_threads = 1;
}

GsidHead *gsid_create(const GsidConfig *cfg, char *err, size_t err_len) {
  auto fail = [&](const std::string &m) -> GsidHead * {
    if (err && err_len) { std::snprintf(err, err_len, "%s", m.c_str()); }
    return nullptr;
  };
  if (!cfg || !cfg->onnx_path || !cfg->ngrams_path || !cfg->labels_path)
    return fail("onnx_path, ngrams_path and labels_path are all required");

  auto h = std::make_unique<GsidHead>();
  std::string e;

  if (!h->vocab.load(cfg->ngrams_path, &e)) return fail(e);
  if (!read_lines(cfg->labels_path, &h->labels, &e)) return fail(e);
  while (!h->labels.empty() && h->labels.back().empty()) h->labels.pop_back();
  if (h->labels.empty()) return fail("labels.txt is empty");

  if (cfg->config_path && *cfg->config_path) {
    std::vector<std::string> lines;
    if (read_lines(cfg->config_path, &lines, &e)) {
      for (const auto &l : lines) {
        const size_t sp = l.find(' ');
        if (sp == std::string::npos) continue;
        const std::string key = l.substr(0, sp), val = l.substr(sp + 1);
        if (key == "ngram_min") h->ngram_min = std::atoi(val.c_str());
        else if (key == "ngram_max") h->ngram_max = std::atoi(val.c_str());
      }
    }
  }
  if (h->ngram_min < 1 || h->ngram_max < h->ngram_min)
    return fail("invalid ngram range in head_config.txt");

  try {
    const int nt = cfg->num_threads > 0 ? cfg->num_threads : 1;
    h->opts.SetIntraOpNumThreads(nt);
    h->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    h->session = std::make_unique<Ort::Session>(h->env, cfg->onnx_path, h->opts);

    Ort::AllocatorWithDefaultOptions alloc;
    if (h->session->GetInputCount() != 2 || h->session->GetOutputCount() != 2)
      return fail("expected a 2-input / 2-output head graph");
    h->in0  = h->session->GetInputNameAllocated(0, alloc).get();
    h->in1  = h->session->GetInputNameAllocated(1, alloc).get();
    h->out0 = h->session->GetOutputNameAllocated(0, alloc).get();
    h->out1 = h->session->GetOutputNameAllocated(1, alloc).get();
  } catch (const Ort::Exception &ex) {
    return fail(std::string("onnxruntime: ") + ex.what());
  }

  return h.release();
}

void gsid_destroy(GsidHead *head) { delete head; }

int gsid_num_languages(const GsidHead *head) {
  return head ? static_cast<int>(head->labels.size()) : 0;
}

const char *gsid_language(const GsidHead *head, int index) {
  if (!head || index < 0 || index >= static_cast<int>(head->labels.size())) return nullptr;
  return head->labels[static_cast<size_t>(index)].c_str();
}

GsidResult gsid_classify(GsidHead *head, const char *ipa) {
  GsidResult r{-1, 0.0f, 0};
  if (!head || !ipa) return r;
  try {
    std::vector<Ort::Value> outs;
    int matched = 0;
    const float *probs = run(head, ipa, &matched, &outs);
    r.num_matched = matched;
    if (!probs) return r;
    const int n = static_cast<int>(head->labels.size());
    int best = 0;
    for (int i = 1; i < n; ++i) if (probs[i] > probs[best]) best = i;
    r.index = best;
    r.confidence = probs[best];
  } catch (const Ort::Exception &) {
    r.index = -1; r.confidence = 0.0f;
  }
  return r;
}

int gsid_classify_probs(GsidHead *head, const char *ipa, float *probs_out) {
  if (!head || !ipa || !probs_out) return 0;
  try {
    std::vector<Ort::Value> outs;
    int matched = 0;
    const float *probs = run(head, ipa, &matched, &outs);
    if (!probs) return 0;
    const int n = static_cast<int>(head->labels.size());
    std::memcpy(probs_out, probs, sizeof(float) * static_cast<size_t>(n));
    return n;
  } catch (const Ort::Exception &) {
    return 0;
  }
}

const char *gsid_version(void) { return GSID_VERSION_STR; }

}  // extern "C"
