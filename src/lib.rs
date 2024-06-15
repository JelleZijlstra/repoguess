use pyo3::prelude::*;

#[pyclass(get_all, frozen)]
struct NameData {
    collection: i32,
    tl_country: i32,
    year: i32,
    single_author: i32,
    authors: std::collections::HashSet<i32>,
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
        single_author: i32,
        authors: std::collections::HashSet<i32>,
        citation_group: i32,
        name_id: i32,
    ) -> Self {
        NameData {
            collection,
            tl_country,
            year,
            single_author,
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
    score_cutoff: f64,
}

#[pymethods]
impl Params {
    #[new]
    fn new(
        country_boost: f64,
        cg_boost: f64,
        author_boost: f64,
        year_factor: f64,
        score_cutoff: f64,
    ) -> Self {
        Params {
            country_boost,
            cg_boost,
            author_boost,
            year_factor,
            score_cutoff,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Params(country_boost={}, cg_boost={}, author_boost={}, year_factor={}, score_cutoff={})",
            self.country_boost, self.cg_boost, self.author_boost, self.year_factor, self.score_cutoff
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
    if (nam1.single_author != 0) && (nam2.single_author != 0) {
        if nam1.single_author == nam2.single_author {
            score *= params.author_boost;
        }
    }
    else if nam1.single_author != 0 {
        if nam2.authors.contains(&nam1.single_author) {
            score *= params.author_boost / (nam2.authors.len() as f64);
        }
    }
    else if nam2.single_author != 0 {
        if nam1.authors.contains(&nam2.single_author) {
            score *= params.author_boost / (nam1.authors.len() as f64);
        }
    }
    else {
        let shared_authors = nam1.authors.intersection(&nam2.authors).count();
        let total_authors = nam1.authors.union(&nam2.authors).count();
        let overlap_prop = (shared_authors as f64) / (total_authors as f64);
        if overlap_prop > 0.0 {
            score *= overlap_prop * params.author_boost;
        }
    }
    let year_difference = (nam1.year - nam2.year).abs();
    let root = (year_difference as f64).sqrt();
    let year_boost = 1.0 - (root / params.year_factor);
    score *= year_boost;
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
fn get_top_choice(data: &NameData, train_data: Vec<Bound<'_, NameData>>, params: &Params, cutoff: f64) -> PyResult<i32> {
    return get_top_choice_impl(data, &train_data, params, cutoff);
}

fn get_top_choice_impl(data: &NameData, train_data: &Vec<Bound<'_, NameData>>, params: &Params, cutoff: f64) -> PyResult<i32> {
    let probs = get_probs_impl(data, train_data, params)?;
    let mut top_choice: i32 = -1;
    let mut top_score: f64 = cutoff;
    for (key, value) in probs.iter() {
        if *value > top_score {
            top_choice = *key;
            top_score = *value;
        }
    }
    return Ok(top_choice);
}

const FALSE_POSITIVE_COST: i32 = 10;

#[pyfunction]
fn evaluate_model(train_data: Vec<Bound<'_, NameData>>, test_data: Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<i32> {
    return evaluate_model_impl(&train_data, &test_data, params);
}

fn evaluate_model_impl(train_data: &Vec<Bound<'_, NameData>>, test_data: &Vec<Bound<'_, NameData>>, params: &Params) -> PyResult<i32> {
    let mut correct: i32 = 0;
    let mut incorrect: i32 = 0;
    for nam in test_data {
        let top_choice = get_top_choice_impl(nam.get(), train_data, params, 0.9)?;
        if top_choice == -1 {
            continue;
        }
        if top_choice == nam.get().collection {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
    return Ok(correct - (incorrect * FALSE_POSITIVE_COST));
}

#[pymodule]
fn typeinfer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_score, m)?)?;
    m.add_function(wrap_pyfunction!(get_probs, m)?)?;
    m.add_function(wrap_pyfunction!(get_top_choice, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_model, m)?)?;
    m.add_class::<NameData>()?;
    m.add_class::<Params>()?;
    Ok(())
}
