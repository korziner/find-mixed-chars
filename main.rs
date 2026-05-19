use clap::{Parser, ValueEnum};
use encoding_rs::{Encoding, WINDOWS_1251, UTF_8};
use encoding_rs_io::DecodeReaderBytesBuilder;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream};
use rayon::prelude::*;

/// Утилита для обнаружения латиницы, цифр и нехарактерных кириллических символов
/// внутри слов, ожидаемо написанных на русском языке.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Обнаруживает инородные символы в кириллических словах с пословным контекстом",
    long_about = "РАЦИОНАЛЬНОСТЬ И ПРИНЦИП РАБОТЫ\n\
                  Исторические OCR-корпуса часто содержат смешанные скрипты из-за визуального сходства символов \
                  (латинские a, c, e, o, p, x, y ↔ кириллические а, с, е, о, р, х, у), ошибок распознавания \
                  шрифтов или типографских артефактов. Наличие инородных символов внутри кириллических токенов \
                  нарушает работу лемматизаторов, частотных анализаторов и нормализаторов регистра.\n\n\
                  ЛОГИКА ДЕТЕКЦИИ:\n\
                  1. Токены разделяются по пробелам, ведущая/замыкающая пунктуация автоматически отбрасывается.\n\
                  2. Токен помечается как 'смешанный', если содержит:\n\
                     - хотя бы одну допустимую русскую букву (А-Яа-я, Ёё, опционально дореформенные Ѣ, Ѳ, І, і);\n\
                     - и хотя бы один 'подозрительный' символ (латиница, цифры, сербские/украинские буквы, греческие омографы).\n\
                  3. Чисто латинские, цифровые или исключительно кириллические токены игнорируются.\n\n\
                  КОНТЕКСТ И ПАРАЛЛЕЛИЗАЦИЯ:\n\
                  - Поиск совпадений выполняется параллельно по всем ядрам CPU через rayon.\n\
                  - Контекст (--context) собирается в СЛОВАХ, а не в строках. При необходимости алгоритм \
                    автоматически заглядывает в предыдущие/последующие строки для набора нужного числа слов.\n\
                  - Цветовой вывод (--color) автоматически отключается при перенаправлении в файл.\n\n\
                  ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:\n\
                  # Пословный контекст 5 слов (по умолчанию), авто-цвет\n\
                  find-mixed-chars -i ocr.txt\n\n\
                  # Контекст 10 слов, принудительные цвета, подробный вывод\n\
                  find-mixed-chars -i ocr.txt -C 10 --color always -v\n\n\
                  # Отключение дореформенных букв, вывод в файл (цвета отключатся)\n\
                  find-mixed-chars -i corpus.txt --no-allow-prereform -o mixed.log\n\n\
                  # Явная кодировка, контекст 0 (только совпадения без окружения)\n\
                  find-mixed-chars -i legacy.txt -e windows-1251 -C 0 -v\n\n\
                  ФОРМАТ ВЫВОДА:\n\
                  [НОМЕР_СТРОКИ:НОМЕР_СЛОВА] | [ИНРОДНЫЕ_СИМВОЛЫ] | КОНТЕКСТ СЛОВ\n\
                  Совпавшее слово выделяется цветом в терминале."
)]
struct Args {
    /// Путь к входному текстовому файлу
    #[arg(short, long)]
    input: String,

    /// Путь к выходному файлу (если не указан, вывод в stdout)
    #[arg(short, long)]
    output: Option<String>,

    /// Включить подробный вывод с логированием этапов обработки
    #[arg(short, long)]
    verbose: bool,

    /// Разрешить дореформенные русские буквы (Ѣ, ѣ, Ѳ, ѳ, І, і). По умолчанию включено.
    #[arg(long, default_value_t = true)]
    allow_prereform: bool,

    /// Кодировка входного файла: utf-8, windows-1251 [default: utf-8]
    #[arg(short, long, default_value = "utf-8")]
    encoding: String,

    /// Количество слов контекста до и после совпадения [default: 5]
    #[arg(short = 'C', long, default_value_t = 5)]
    context: usize,

    /// Режим цветного вывода: auto, always, never [default: auto]
    #[arg(long, value_enum, default_value_t = ColorMode::Auto)]
    color: ColorMode,
}

#[derive(ValueEnum, Clone, Debug)]
enum ColorMode { Auto, Always, Never }

fn is_valid_russian(c: char, allow_prereform: bool) -> bool {
    match c {
        'А'..='я' | 'Ё' | 'ё' => true,
        'Ѣ' | 'ѣ' | 'Ѳ' | 'ѳ' | 'І' | 'і' if allow_prereform => true,
        '-' | '\'' | '‘' | '’' | '`' => true,
        _ => false,
    }
}

fn is_cyrillic_block(c: char) -> bool {
    matches!(
        c,
        '\u{0400}'..='\u{04FF}' |
        '\u{0500}'..='\u{052F}' |
        '\u{2DE0}'..='\u{2DFF}' |
        '\u{A640}'..='\u{A69F}'
    )
}

fn is_suspicious(c: char) -> bool {
    if c.is_ascii_alphanumeric() {
        return true;
    }
    if matches!(c, 'ј' | 'љ' | 'њ' | 'ћ' | 'ђ' | 'џ' | 'Ј' | 'Љ' | 'Њ' | 'Ћ' | 'Ђ' | 'Џ' | 'є' | 'Є' | 'ґ' | 'Ґ' | 'ї' | 'Ї' | 'ў' | 'Ў') {
        return true;
    }
    if is_cyrillic_block(c) && !matches!(c, 'А'..='я' | 'Ё' | 'ё' | 'Ѣ' | 'ѣ' | 'Ѳ' | 'ѳ' | 'І' | 'і') {
        return true;
    }
    if matches!(c, 'ο' | 'Ο' | 'σ' | 'Σ') {
        return true;
    }
    false
}

fn resolve_encoding(name: &str) -> Result<&'static Encoding, io::Error> {
    match name.to_lowercase().as_str() {
        "utf-8" | "utf8" => Ok(UTF_8),
        "windows-1251" | "cp1251" | "win1251" => Ok(WINDOWS_1251),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Неподдерживаемая кодировка: '{}'. Допустимые: utf-8, windows-1251", name),
        )),
    }
}

fn extract_suspicious_tokens(line: &str, allow_prereform: bool) -> Option<(String, String)> {
    let mut all_foreign = HashSet::new();
    let mut has_russian = false;
    let mut has_suspicious = false;

    for token in line.split_whitespace() {
        let core: String = token.chars().filter(|c| c.is_alphanumeric() || matches!(c, '-' | '\'' | '‘' | '’' | '`')).collect();
        if core.is_empty() { continue; }

        let mut line_has_russian = false;
        let mut line_foreign = HashSet::new();

        for c in core.chars() {
            if is_valid_russian(c, allow_prereform) {
                line_has_russian = true;
            } else if is_suspicious(c) {
                line_foreign.insert(c);
            }
        }

        if line_has_russian && !line_foreign.is_empty() {
            has_russian = true;
            has_suspicious = true;
            all_foreign.extend(line_foreign);
        }
    }

    if has_russian && has_suspicious {
        let foreign_str: String = all_foreign.into_iter().collect();
        // Возвращаем инородные символы и всю строку для последующего поиска контекста
        Some((foreign_str, line.trim().to_string()))
    } else {
        None
    }
}

/// Извлекает N слов до и после целевого токена, корректно пересекая границы строк.
fn build_word_context(
    lines: &[String],
    line_idx: usize,
    target_line: &str,
    n_words: usize
) -> String {
    if n_words == 0 {
        return target_line.to_string();
    }

    // Определяем целевое слово из строки совпадения (первое подозрительное)
    let target_word = target_line.split_whitespace()
        .find(|t| {
            let core: String = t.chars().filter(|c| c.is_alphanumeric() || matches!(c, '-' | '\'' | '‘' | '’' | '`')).collect();
            !core.is_empty() && core.chars().any(is_suspicious)
        })
        .unwrap_or("");

    // Собираем слова из окна строк
    let start_line = line_idx.saturating_sub(3); // эмпирический буфер
    let end_line = (line_idx + 3).min(lines.len() - 1);

    let mut words: Vec<&str> = Vec::new();
    let mut target_global_idx = None;

    for i in start_line..=end_line {
        for word in lines[i].split_whitespace() {
            words.push(word);
            if target_global_idx.is_none() && word == target_word {
                target_global_idx = Some(words.len() - 1);
            }
        }
    }

    if let Some(idx) = target_global_idx {
        let start = idx.saturating_sub(n_words);
        let end = (idx + n_words + 1).min(words.len());
        words[start..end].join(" ")
    } else {
        // Фоллбэк: если точное слово не найдено (редко), возвращаем оригинальную строку
        target_line.to_string()
    }
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if args.verbose {
        eprintln!(
            "Запуск: вход='{}', выход='{:?}', слов_контекста={}, цвет={:?}, потоков={}",
            args.input, args.output, args.context, args.color, rayon::current_num_threads()
        );
    }

    let encoding = resolve_encoding(&args.encoding)?;
    let file = File::open(&args.input).map_err(|e| {
        io::Error::new(e.kind(), format!("Не удалось открыть '{}': {}", args.input, e))
    })?;

    let decoder = DecodeReaderBytesBuilder::new()
        .encoding(Some(encoding))
        .utf8_passthru(true)
        .build(file);
    let reader = BufReader::with_capacity(1024 * 1024, decoder);

    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
    let total_lines = lines.len();

    if args.verbose {
        eprintln!("Загружено строк: {}. Запуск параллельного анализа...", total_lines);
    }

    let matches: Vec<(usize, String, String)> = lines.par_iter()
        .enumerate()
        .filter_map(|(i, line)| {
            extract_suspicious_tokens(line, args.allow_prereform)
                .map(|res| (i, res.0, res.1))
        })
        .collect();

    let match_count = matches.len();
    if args.verbose {
        eprintln!("Обнаружено совпадений: {}. Извлечение пословного контекста...", match_count);
    }

    if match_count == 0 {
        if args.verbose { eprintln!("Совпадений не найдено."); }
        return Ok(());
    }

    let mut lines_to_print: BTreeSet<usize> = BTreeSet::new();
    let mut match_data: HashMap<usize, (String, String)> = HashMap::new();

    for &(idx, ref foreign, ref line) in &matches {
        lines_to_print.insert(idx);
        match_data.insert(idx, (foreign.clone(), line.clone()));
    }

    let color_choice = match args.color {
        ColorMode::Auto => if args.output.is_some() { ColorChoice::Never } else { ColorChoice::Auto },
        ColorMode::Always => ColorChoice::Always,
        ColorMode::Never => ColorChoice::Never,
    };

    let mut color_spec = ColorSpec::new();
    color_spec.set_fg(Some(Color::Red)).set_bold(true);

    // Подготовка буфера вывода
    let mut out: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(BufWriter::with_capacity(1024 * 1024, File::create(path)?)),
        None => Box::new(BufWriter::with_capacity(1024 * 1024, io::stdout())),
    };

    for &line_idx in &lines_to_print {
        let (foreign_chars, full_line) = &match_data[&line_idx];
        let context = build_word_context(&lines, line_idx, full_line, args.context);

        if color_choice == ColorChoice::Never {
            // Файловый/безцветный режим
            writeln!(out, "{:<4} | [{}] | {}", line_idx + 1, foreign_chars, context)?;
        } else {
            // Терминальный режим с цветами
            use termcolor::WriteColor;
            let mut term_out = StandardStream::stdout(color_choice);
            write!(term_out, "{:<4} | [{}] | ", line_idx + 1, foreign_chars)?;
            for c in context.chars() {
                if is_suspicious(c) {
                    term_out.set_color(&color_spec)?;
                    write!(term_out, "{}", c)?;
                    term_out.reset()?;
                } else {
                    write!(term_out, "{}", c)?;
                }
            }
            writeln!(term_out)?;
        }
    }

    out.flush()?;
    if args.verbose {
        eprintln!("Обработка завершена. Результат сохранён/выведен.");
    }
    Ok(())
}
