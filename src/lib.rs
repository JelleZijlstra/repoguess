use pyo3::prelude::*;

#[pyclass(get_all, frozen)]
struct NameData {
    collection: i32,
    tl_country: i32,
    year: i32,
    authors: Vec<i32>,
    citation_group: i32,
    name_id: i32,
}

#[pymethods]
impl NameData {
    #[new]
    fn new(
        collection: i32,
        tl_country: i32,
        year: i32,
        authors: Vec<i32>,
        citation_group: i32,
        name_id: i32,
    ) -> Self {
        NameData {
            collection,
            tl_country,
            year,
            authors,
            citation_group,
            name_id,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "NameData(collection={}, tl_country={}, year={}, authors={:?}, citation_group={}, name_id={})",
            self.collection, self.tl_country, self.year, self.authors, self.citation_group, self.name_id
        ))
    }
}

#[pyclass(get_all, frozen)]
struct Params {
    country_boost: f64,
    cg_boost: f64,
    author_boost: f64,
    year_factor: f64,
    year_boost: f64,
    score_cutoff: f64,
    probability_cutoff: f64,
}

#[pymethods]
impl Params {
    #[new]
    fn new(
        country_boost: f64,
        cg_boost: f64,
        author_boost: f64,
        year_factor: f64,
        year_boost: f64,
        score_cutoff: f64,
        probability_cutoff: f64,
    ) -> Self {
        Params {
            country_boost,
            cg_boost,
            author_boost,
            year_factor,
            year_boost,
            score_cutoff,
            probability_cutoff,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Params(country_boost={:.3}, cg_boost={:.3}, author_boost={:.3}, year_factor={:.3}, year_boost={:3}, score_cutoff={:.3}, probability_cutoff={:.3})",
            self.country_boost, self.cg_boost, self.author_boost, self.year_factor, self.year_boost, self.score_cutoff, self.probability_cutoff
        ))
    }
}

#[pyfunction]
fn get_score(nam1: &NameData, nam2: &NameData, params: &Params) -> PyResult<f64> {
    if nam1.name_id == nam2.name_id {
        return Ok(0.0);
    }
    let mut score: f64 = 1.0;
    if nam1.tl_country == nam2.tl_country {
        score *= params.country_boost;
    }
    if nam1.citation_group == nam2.citation_group {
        score *= params.cg_boost;
    }
    if nam1.authors == nam2.authors {
        score *= params.author_boost;
    }
    else {
        // Faster than using sets since usually it's a small number of authors
        let mut shared_authors = 0;
        for author in &nam1.authors {
            if nam2.authors.contains(&author) {
                shared_authors += 1;
            }
        }
        let total_authors = nam1.authors.len() + nam2.authors.len() - shared_authors;
        let overlap_prop = (shared_authors as f64) / (total_authors as f64);
        if overlap_prop > 0.0 {
            score *= overlap_prop * params.author_boost;
        }
    }
    let year_difference = (nam1.year - nam2.year).abs();
    score *= (1.0 / params.year_factor.powf(year_difference as f64)) * params.year_boost;
    return Ok(score);
}

#[pyfunction]
fn get_probs(data: &NameData, train_data: Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<std::collections::HashMap<i32, f64>> {
    return get_probs_impl(data, &train_data, params);
}

fn get_probs_impl(data: &NameData, train_data: &Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<std::collections::HashMap<i32, f64>> {
    let mut scores: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
    let mut highest_score: f64 = 1.0;
    for train_datum in train_data {
        let score = get_score(data, train_datum.get(), params)?;
        if score > params.score_cutoff {
            *scores.entry(train_datum.get().collection).or_insert(0.0) += score;
        }
        if score > highest_score {
            highest_score = score;
        }
    }
    *scores.entry(0).or_insert(0.0) += highest_score;
    let total_score: f64 = scores.values().sum();
    let mut result: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
    for (key, value) in scores.iter() {
        *result.entry(*key).or_insert(0.0) = value / total_score;
    }
    return Ok(result);
}

#[pyfunction]
fn get_top_choice(data: &NameData, train_data: Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<Option<(i32, f64)>> {
    let probs = get_probs_impl(data, &train_data, params)?;
    let mut top_choice: i32 = -1;
    let mut top_score: f64 = params.probability_cutoff;
    for (key, value) in probs.iter() {
        if *value > top_score {
            top_choice = *key;
            top_score = *value;
        }
    }
    if top_choice == -1 {
        return Ok(None);
    }
    return Ok(Some((top_choice, top_score)));
}

fn get_top_choice_impl(data: &NameData, train_data: &Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<i32> {
    let probs = get_probs_impl(data, train_data, params)?;
    let mut top_choice: i32 = -1;
    let mut top_score: f64 = params.probability_cutoff;
    for (key, value) in probs.iter() {
        if *value > top_score {
            top_choice = *key;
            top_score = *value;
        }
    }
    return Ok(top_choice);
}

const FALSE_POSITIVE_COST: i32 = 10;

#[pyclass(get_all, frozen)]
struct ScoreInfo {
    score: i32,
    correct: i32,
    incorrect: i32,
    no_value: i32,
}

#[pymethods]
impl ScoreInfo {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ScoreInfo(score={}, correct={}, incorrect={}, no_value={})",
            self.score, self.correct, self.incorrect, self.no_value
        ))
    }
}

#[pyfunction]
fn evaluate_model(train_data: Vec<Bound<'_, NameData>>, test_data: Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<ScoreInfo> {
    return evaluate_model_impl(&train_data, &test_data, params);
}

fn evaluate_model_impl(train_data: &Vec<Bound<'_, NameData>>, test_data: &Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<ScoreInfo> {
    let mut correct: i32 = 0;
    let mut incorrect: i32 = 0;
    let mut no_value: i32 = 0;
    for nam in test_data {
        let top_choice = get_top_choice_impl(nam.get(), train_data, params)?;
        if top_choice == -1 {
            no_value += 1;
            continue;
        }
        if top_choice == nam.get().collection {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
    let score = correct - (incorrect * FALSE_POSITIVE_COST);
    return Ok(
        ScoreInfo {
            score,
            correct,
            incorrect,
            no_value,
        }
    );
}

#[pymodule]
fn typeinfer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_score, m)?)?;
    m.add_function(wrap_pyfunction!(get_probs, m)?)?;
    m.add_function(wrap_pyfunction!(get_top_choice, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_model, m)?)?;
    m.add_class::<NameData>()?;
    m.add_class::<Params>()?;
    m.add_class::<ScoreInfo>()?;
    Ok(())
}
