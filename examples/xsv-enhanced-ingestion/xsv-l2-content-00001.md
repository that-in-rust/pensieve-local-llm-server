# Export Row 1

## Metadata

- **Row Index**: 1
- **Total Rows**: 1
- **Exported At**: 2025-09-28T07:48:24.185048+00:00
- **Query Time**: 7ms

## Data

**L2 Window Content**: 

```
use csv;

use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Concatenates CSV data by column or by row.

When concatenating by column, the columns will be written in the same order as
the inputs given. The number of rows in the result is always equivalent to to
the minimum number of rows across all given CSV data. (This behavior can be
reversed with the '--pad' flag.)

When concatenating by row, all CSV data must have the same number of columns.
If you need to rearrange the columns or fix the lengths of records, use the
'select' or 'fixlengths' commands. Also, only the headers of the *first* CSV
data given are used. Headers in subsequent inputs are ignored. (This behavior
can be disabled with --no-headers.)

Usage:
    xsv cat rows    [options] [<input>...]
    xsv cat columns [options] [<input>...]
    xsv cat --help

cat options:
    -p, --pad              When concatenating columns, this flag will cause
                           all records to appear. It will pad each row if
                           other CSV data isn't long enough.

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will NOT be interpreted
                           as column names. Note that this has no effect when
                           concatenating columns.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    cmd_rows: bool,
    cmd_columns: bool,
    arg_input: Vec<String>,
    flag_pad: bool,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    if args.cmd_rows {
        args.cat_rows()
    } else if args.cmd_columns {
        args.cat_columns()
    } else {
        unreachable!();
    }
}

impl Args {
    fn configs(&self) -> CliResult<Vec<Config>> {
        util::many_configs(&*self.arg_input,
                           self.flag_delimiter,
                           self.flag_no_headers)
             .map_err(From::from)
    }

    fn cat_rows(&self) -> CliResult<()> {
        let mut row = csv::ByteRecord::new();
        let mut wtr = Config::new(&self.flag_output).writer()?;
        for (i, conf) in self.configs()?.into_iter().enumerate() {
            let mut rdr = conf.reader()?;
            if i == 0 {
                conf.write_headers(&mut rdr, &mut wtr)?;
            }
            while rdr.read_byte_record(&mut row)? {
                wtr.write_byte_record(&row)?;
            }
        }
        wtr.flush().map_err(From::from)
    }

    fn cat_columns(&self) -> CliResult<()> {
        let mut wtr = Config::new(&self.flag_output).writer()?;
        let mut rdrs = self.configs()?
            .into_iter()
            .map(|conf| conf.no_headers(true).reader())
            .collect::<Result<Vec<_>, _>>()?;

        // Find the lengths of each record. If a length varies, then an error
        // will occur so we can rely on the first length being the correct one.
        let mut lengths = vec![];
        for rdr in &mut rdrs {
            lengths.push(rdr.byte_headers()?.len());
        }

        let mut iters = rdrs.iter_mut()
                            .map(|rdr| rdr.byte_records())
                            .collect::<Vec<_>>();
        'OUTER: loop {
            let mut record = csv::ByteRecord::new();
            let mut num_done = 0;
            for (iter, &len) in iters.iter_mut().zip(lengths.iter()) {
                match iter.next() {
                    None => {
                        num_done += 1;
                        if self.flag_pad {
                            for _ in 0..len {
                                record.push_field(b"");
                            }
                        } else {
                            break 'OUTER;
                        }
                    }
                    Some(Err(err)) => return fail!(err),
                    Some(Ok(next)) => record.extend(&next),
                }
            }
            // Only needed when `--pad` is set.
            // When not set, the OUTER loop breaks when the shortest iterator
            // is exhausted.
            if num_done >= iters.len() {
                break 'OUTER;
            }
            wtr.write_byte_record(&record)?;
        }
        wtr.flush().map_err(From::from)
    }
}

--- MODULE SEPARATOR ---
use csv;

use CliResult;
use config::{Delimiter, Config};
use util;

static USAGE: &'static str = "
Prints a count of the number of records in the CSV data.

Note that the count will not include the header row (unless --no-headers is
given).

Usage:
    xsv count [options] [<input>]

Common options:
    -h, --help             Display this message
    -n, --no-headers       When set, the first row will not be included in
                           the count.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let conf = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(args.flag_no_headers);

    let count =
        match conf.indexed()? {
            Some(idx) => idx.count(),
            None => {
                let mut rdr = conf.reader()?;
                let mut count = 0u64;
                let mut record = csv::ByteRecord::new();
                while rdr.read_byte_record(&mut record)? {
                    count += 1;
                }
                count
            }
        };
    Ok(println!("{}", count))
}

--- MODULE SEPARATOR ---
use std::cmp;

use csv;

use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Transforms CSV data so that all records have the same length. The length is
the length of the longest record in the data (not counting trailing empty fields,
but at least 1). Records with smaller lengths are padded with empty fields.

This requires two complete scans of the CSV data: one for determining the
record size and one for the actual transform. Because of this, the input
given must be a file and not stdin.

Alternatively, if --length is set, then all records are forced to that length.
This requires a single pass and can be done with stdin.

Usage:
    xsv fixlengths [options] [<input>]

fixlengths options:
    -l, --length <arg>     Forcefully set the length of each record. If a
                           record is not the size given, then it is truncated
                           or expanded as appropriate.

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_length: Option<usize>,
    flag_output: Option<String>,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let config = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(true)
        .flexible(true);
    let length = match args.flag_length {
        Some(length) => {
            if length == 0 {
                return fail!("Length must be greater than 0.");
            }
            length
        }
        None => {
            if config.is_std() {
                return fail!("<stdin> cannot be used in this command. \
                              Please specify a file path.");
            }
            let mut maxlen = 0usize;
            let mut rdr = config.reader()?;
            let mut record = csv::ByteRecord::new();
            while rdr.read_byte_record(&mut record)? {
                let mut index = 0;
                let mut nonempty_count = 0;
                for field in &record {
                    index += 1;
                    if index == 1 || !field.is_empty() {
                        nonempty_count = index;
                    }
                }
                maxlen = cmp::max(maxlen, nonempty_count);
            }
            maxlen
        }
    };

    let mut rdr = config.reader()?;
    let mut wtr = Config::new(&args.flag_output).writer()?;
    for r in rdr.byte_records() {
        let mut r = r?;
        if length >= r.len() {
            for _ in r.len()..length {
                r.push_field(b"");
            }
        } else {
            r.truncate(length);
        }
        wtr.write_byte_record(&r)?;
    }
    wtr.flush()?;
    Ok(())
}

--- MODULE SEPARATOR ---
use std::borrow::Cow;
use std::io::{self, Write};

use tabwriter::TabWriter;

use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Prints flattened records such that fields are labeled separated by a new line.
This mode is particularly useful for viewing one record at a time. Each
record is separated by a special '#' character (on a line by itself), which
can be changed with the --separator flag.

There is also a condensed view (-c or --condense) that will shorten the
contents of each field to provide a summary view.

Usage:
    xsv flatten [options] [<input>]

flatten options:
    -c, --condense <arg>  Limits the length of each field to the value
                           specified. If the field is UTF-8 encoded, then
                           <arg> refers to the number of code points.
                           Otherwise, it refers to the number of bytes.
    -s, --separator <arg>  A string of characters to write after each record.
                           When non-empty, a new line is automatically
                           appended to the separator.
                           [default: #]

Common options:
    -h, --help             Display this message
    -n, --no-headers       When set, the first row will not be interpreted
                           as headers. When set, the name of each field
                           will be its index.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_condense: Option<usize>,
    flag_separator: String,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(args.flag_no_headers);
    let mut rdr = rconfig.reader()?;
    let headers = rdr.byte_headers()?.clone();

    let mut wtr = TabWriter::new(io::stdout());
    let mut first = true;
    for r in rdr.byte_records() {
        if !first && !args.flag_separator.is_empty() {
            writeln!(&mut wtr, "{}", args.flag_separator)?;
        }
        first = false;
        let r = r?;
        for (i, (header, field)) in headers.iter().zip(&r).enumerate() {
            if rconfig.no_headers {
                write!(&mut wtr, "{}", i)?;
            } else {
                wtr.write_all(&header)?;
            }
            wtr.write_all(b"\t")?;
            wtr.write_all(&*util::condense(
                Cow::Borrowed(&*field), args.flag_condense))?;
            wtr.write_all(b"\n")?;
        }
    }
    wtr.flush()?;
    Ok(())
}

--- MODULE SEPARATOR ---
use csv;

use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Formats CSV data with a custom delimiter or CRLF line endings.

Generally, all commands in xsv output CSV data in a default format, which is
the same as the default format for reading CSV data. This makes it easy to
pipe multiple xsv commands together. However, you may want the final result to
have a specific delimiter or record separator, and this is where 'xsv fmt' is
useful.

Usage:
    xsv fmt [options] [<input>]

fmt options:
    -t, --out-delimiter <arg>  The field delimiter for writing CSV data.
                               [default: ,]
    --crlf                     Use '\\r\\n' line endings in the output.
    --ascii                    Use ASCII field and record separators.
    --quote <arg>              The quote character to use. [default: \"]
    --quote-always             Put quotes around every value.
    --escape <arg>             The escape character to use. When not specified,
                               quotes are escaped by doubling them.

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_out_delimiter: Option<Delimiter>,
    flag_crlf: bool,
    flag_ascii: bool,
    flag_output: Option<String>,
    flag_delimiter: Option<Delimiter>,
    flag_quote: Delimiter,
    flag_quote_always: bool,
    flag_escape: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;

    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(true);
    let mut wconfig = Config::new(&args.flag_output)
        .delimiter(args.flag_out_delimiter)
        .crlf(args.flag_crlf);

    if args.flag_ascii {
        wconfig = wconfig
            .delimiter(Some(Delimiter(b'\x1f')))
            .terminator(csv::Terminator::Any(b'\x1e'));
    }
    if args.flag_quote_always {
        wconfig = wconfig.quote_style(csv::QuoteStyle::Always);
    }
    if let Some(escape) = args.flag_escape {
        wconfig = wconfig.escape(Some(escape.as_byte())).double_quote(false);
    }
    wconfig = wconfig.quote(args.flag_quote.as_byte());


    let mut rdr = rconfig.reader()?;
    let mut wtr = wconfig.writer()?;
    let mut r = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut r)? {
        wtr.write_byte_record(&r)?;
    }
    wtr.flush()?;
    Ok(())
}

--- MODULE SEPARATOR ---
use std::fs;
use std::io;

use channel;
use csv;
use stats::{Frequencies, merge_all};
use threadpool::ThreadPool;

use CliResult;
use config::{Config, Delimiter};
use index::Indexed;
use select::{SelectColumns, Selection};
use util;

static USAGE: &'static str = "
Compute a frequency table on CSV data.

The frequency table is formatted as CSV data:

    field,value,count

By default, there is a row for the N most frequent values for each field in the
data. The order and number of values can be tweaked with --asc and --limit,
respectively.

Since this computes an exact frequency table, memory proportional to the
cardinality of each column is required.

Usage:
    xsv frequency [options] [<input>]

frequency options:
    -s, --select <arg>     Select a subset of columns to compute frequencies
                           for. See 'xsv select --help' for the format
                           details. This is provided here because piping 'xsv
                           select' into 'xsv frequency' will disable the use
                           of indexing.
    -l, --limit <arg>      Limit the frequency table to the N most common
                           items. Set to '0' to disable a limit.
                           [default: 10]
    -a, --asc              Sort the frequency tables in ascending order by
                           count. The default is descending order.
    --no-nulls             Don't include NULLs in the frequency table.
    -j, --jobs <arg>       The number of jobs to run in parallel.
                           This works better when the given CSV data has
                           an index already created. Note that a file handle
                           is opened for each job.
                           When set to '0', the number of jobs is set to the
                           number of CPUs detected.
                           [default: 0]

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will NOT be included
                           in the frequency table. Additionally, the 'field'
                           column will be 1-based indices instead of header
                           names.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Clone, Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_select: SelectColumns,
    flag_limit: usize,
    flag_asc: bool,
    flag_no_nulls: bool,
    flag_jobs: usize,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let rconfig = args.rconfig();

    let mut wtr = Config::new(&args.flag_output).writer()?;
    let (headers, tables) = match args.rconfig().indexed()? {
        Some(ref mut idx) if args.njobs() > 1 => args.parallel_ftables(idx),
        _ => args.sequential_ftables(),
    }?;

    wtr.write_record(vec!["field", "value", "count"])?;
    let head_ftables = headers.into_iter().zip(tables.into_iter());
    for (i, (header, ftab)) in head_ftables.enumerate() {
        let mut header = header.to_vec();
        if rconfig.no_headers {
            header = (i+1).to_string().into_bytes();
        }
        for (value, count) in args.counts(&ftab).into_iter() {
            let count = count.to_string();
            let row = vec![&*header, &*value, count.as_bytes()];
            wtr.write_record(row)?;
        }
    }
    Ok(())
}

type ByteString = Vec<u8>;
type Headers = csv::ByteRecord;
type FTable = Frequencies<Vec<u8>>;
type FTables = Vec<Frequencies<Vec<u8>>>;

impl Args {
    fn rconfig(&self) -> Config {
        Config::new(&self.arg_input)
            .delimiter(self.flag_delimiter)
            .no_headers(self.flag_no_headers)
            .select(self.flag_select.clone())
    }

    fn counts(&self, ftab: &FTable) -> Vec<(ByteString, u64)> {
        let mut counts = if self.flag_asc {
            ftab.least_frequent()
        } else {
            ftab.most_frequent()
        };
        if self.flag_limit > 0 {
            counts = counts.into_iter().take(self.flag_limit).collect();
        }
        counts.into_iter().map(|(bs, c)| {
            if b"" == &**bs {
                (b"(NULL)"[..].to_vec(), c)
            } else {
                (bs.clone(), c)
            }
        }).collect()
    }

    fn sequential_ftables(&self) -> CliResult<(Headers, FTables)> {
        let mut rdr = self.rconfig().reader()?;
        let (headers, sel) = self.sel_headers(&mut rdr)?;
        Ok((headers, self.ftables(&sel, rdr.byte_records())?))
    }

    fn parallel_ftables(&self, idx: &mut Indexed<fs::File, fs::File>)
                       -> CliResult<(Headers, FTables)> {
        let mut rdr = self.rconfig().reader()?;
        let (headers, sel) = self.sel_headers(&mut rdr)?;

        if idx.count() == 0 {
            return Ok((headers, vec![]));
        }

        let chunk_size = util::chunk_size(idx.count() as usize, self.njobs());
        let nchunks = util::num_of_chunks(idx.count() as usize, chunk_size);

        let pool = ThreadPool::new(self.njobs());
        let (send, recv) = channel::bounded(0);
        for i in 0..nchunks {
            let (send, args, sel) = (send.clone(), self.clone(), sel.clone());
            pool.execute(move || {
                let mut idx = args.rconfig().indexed().unwrap().unwrap();
                idx.seek((i * chunk_size) as u64).unwrap();
                let it = idx.byte_records().take(chunk_size);
                send.send(args.ftables(&sel, it).unwrap());
            });
        }
        drop(send);
        Ok((headers, merge_all(recv).unwrap()))
    }

    fn ftables<I>(&self, sel: &Selection, it: I) -> CliResult<FTables>
            where I: Iterator<Item=csv::Result<csv::ByteRecord>> {
        let null = &b""[..].to_vec();
        let nsel = sel.normal();
        let mut tabs: Vec<_> =
            (0..nsel.len()).map(|_| Frequencies::new()).collect();
        for row in it {
            let row = row?;
            for (i, field) in nsel.select(row.into_iter()).enumerate() {
                let field = trim(field.to_vec());
                if !field.is_empty() {
                    tabs[i].add(field);
                } else {
                    if !self.flag_no_nulls {
                        tabs[i].add(null.clone());
                    }
                }
            }
        }
        Ok(tabs)
    }

    fn sel_headers<R: io::Read>(&self, rdr: &mut csv::Reader<R>)
                  -> CliResult<(csv::ByteRecord, Selection)> {
        let headers = rdr.byte_headers()?;
        let sel = self.rconfig().selection(headers)?;
        Ok((sel.select(headers).map(|h| h.to_vec()).collect(), sel))
    }

    fn njobs(&self) -> usize {
        if self.flag_jobs == 0 { util::num_cpus() } else { self.flag_jobs }
    }
}

fn trim(bs: ByteString) -> ByteString {
    match String::from_utf8(bs) {
        Ok(s) => s.trim().as_bytes().to_vec(),
        Err(bs) => bs.into_bytes(),
    }
}

--- MODULE SEPARATOR ---
use std::io;

use tabwriter::TabWriter;

use CliResult;
use config::Delimiter;
use util;

static USAGE: &'static str = "
Prints the fields of the first row in the CSV data.

These names can be used in commands like 'select' to refer to columns in the
CSV data.

Note that multiple CSV files may be given to this command. This is useful with
the --intersect flag.

Usage:
    xsv headers [options] [<input>...]

headers options:
    -j, --just-names       Only show the header names (hide column index).
                           This is automatically enabled if more than one
                           input is given.
    --intersect            Shows the intersection of all headers in all of
                           the inputs given.

Common options:
    -h, --help             Display this message
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Vec<String>,
    flag_just_names: bool,
    flag_intersect: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let configs = util::many_configs(
        &*args.arg_input, args.flag_delimiter, true)?;

    let num_inputs = configs.len();
    let mut headers: Vec<Vec<u8>> = vec![];
    for conf in configs.into_iter() {
        let mut rdr = conf.reader()?;
        for header in rdr.byte_headers()?.iter() {
            if !args.flag_intersect
                || !headers.iter().any(|h| &**h == header)
            {
                headers.push(header.to_vec());
            }
        }
    }

    let mut wtr: Box<io::Write> =
        if args.flag_just_names {
            Box::new(io::stdout())
        } else {
            Box::new(TabWriter::new(io::stdout()))
        };
    for (i, header) in headers.into_iter().enumerate() {
        if num_inputs == 1 && !args.flag_just_names {
            write!(&mut wtr, "{}\t", i+1)?;
        }
        wtr.write_all(&header)?;
        wtr.write_all(b"\n")?;
    }
    wtr.flush()?;
    Ok(())
}

--- MODULE SEPARATOR ---
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use csv_index::RandomAccessSimple;

use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Creates an index of the given CSV data, which can make other operations like
slicing, splitting and gathering statistics much faster.

Note that this does not accept CSV data on stdin. You must give a file
path. The index is created at 'path/to/input.csv.idx'. The index will be
automatically used by commands that can benefit from it. If the original CSV
data changes after the index is made, commands that try to use it will result
in an error (you have to regenerate the index before it can be used again).

Usage:
    xsv index [options] <input>
    xsv index --help

index options:
    -o, --output <file>    Write index to <file> instead of <input>.idx.
                           Generally, this is not currently useful because
                           the only way to use an index is if it is specially
                           named <input>.idx.

Common options:
    -h, --help             Display this message
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: String,
    flag_output: Option<String>,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;

    let pidx = match args.flag_output {
        None => util::idx_path(&Path::new(&args.arg_input)),
        Some(p) => PathBuf::from(&p),
    };

    let rconfig = Config::new(&Some(args.arg_input))
                         .delimiter(args.flag_delimiter);
    let mut rdr = rconfig.reader_file()?;
    let mut wtr = io::BufWriter::new(fs::File::create(&pidx)?);
    RandomAccessSimple::create(&mut rdr, &mut wtr)?;
    Ok(())
}

--- MODULE SEPARATOR ---
use csv;

use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Read CSV data with special quoting rules.

Generally, all xsv commands support basic options like specifying the delimiter
used in CSV data. This does not cover all possible types of CSV data. For
example, some CSV files don't use '\"' for quotes or use different escaping
styles.

Usage:
    xsv input [options] [<input>]

input options:
    --quote <arg>          The quote character to use. [default: \"]
    --escape <arg>         The escape character to use. When not specified,
                           quotes are escaped by doubling them.
    --no-quoting           Disable quoting completely.

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_output: Option<String>,
    flag_delimiter: Option<Delimiter>,
    flag_quote: Delimiter,
    flag_escape: Option<Delimiter>,
    flag_no_quoting: bool,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let mut rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(true)
        .quote(args.flag_quote.as_byte());
    let wconfig = Config::new(&args.flag_output);

    if let Some(escape) = args.flag_escape {
        rconfig = rconfig.escape(Some(escape.as_byte())).double_quote(false);
    }
    if args.flag_no_quoting {
        rconfig = rconfig.quoting(false);
    }

    let mut rdr = rconfig.reader()?;
    let mut wtr = wconfig.writer()?;
    let mut row = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut row)? {
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

--- MODULE SEPARATOR ---
use std::collections::hash_map::{HashMap, Entry};
use std::fmt;
use std::fs;
use std::io;
use std::iter::repeat;
use std::str;

use byteorder::{WriteBytesExt, BigEndian};
use csv;

use CliResult;
use config::{Config, Delimiter};
use index::Indexed;
use select::{SelectColumns, Selection};
use util;

static USAGE: &'static str = "
Joins two sets of CSV data on the specified columns.

The default join operation is an 'inner' join. This corresponds to the
intersection of rows on the keys specified.

Joins are always done by ignoring leading and trailing whitespace. By default,
joins are done case sensitively, but this can be disabled with the --no-case
flag.

The columns arguments specify the columns to join for each input. Columns can
be referenced by name or index, starting at 1. Specify multiple columns by
separating them with a comma. Specify a range of columns with `-`. Both
columns1 and columns2 must specify exactly the same number of columns.
(See 'xsv select --help' for the full syntax.)

Usage:
    xsv join [options] <columns1> <input1> <columns2> <input2>
    xsv join --help

join options:
    --no-case              When set, joins are done case insensitively.
    --left                 Do a 'left outer' join. This returns all rows in
                           first CSV data set, including rows with no
                           corresponding row in the second data set. When no
                           corresponding row exists, it is padded out with
                           empty fields.
    --right                Do a 'right outer' join. This returns all rows in
                           second CSV data set, including rows with no
                           corresponding row in the first data set. When no
                           corresponding row exists, it is padded out with
                           empty fields. (This is the reverse of 'outer left'.)
    --full                 Do a 'full outer' join. This returns all rows in
                           both data sets with matching records joined. If
                           there is no match, the missing side will be padded
                           out with empty fields. (This is the combination of
                           'outer left' and 'outer right'.)
    --cross                USE WITH CAUTION.
                           This returns the cartesian product of the CSV
                           data sets given. The number of rows return is
                           equal to N * M, where N and M correspond to the
                           number of rows in the given data sets, respectively.
    --nulls                When set, joins will work on empty fields.
                           Otherwise, empty fields are completely ignored.
                           (In fact, any row that has an empty field in the
                           key specified is ignored.)

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will not be interpreted
                           as headers. (i.e., They are not searched, analyzed,
                           sliced, etc.)
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

type ByteString = Vec<u8>;

#[derive(Deserialize)]
struct Args {
    arg_columns1: SelectColumns,
    arg_input1: String,
    arg_columns2: SelectColumns,
    arg_input2: String,
    flag_left: bool,
    flag_right: bool,
    flag_full: bool,
    flag_cross: bool,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_no_case: bool,
    flag_nulls: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let mut state = args.new_io_state()?;
    match (
        args.flag_left,
        args.flag_right,
        args.flag_full,
        args.flag_cross,
    ) {
        (true, false, false, false) => {
            state.write_headers()?;
            state.outer_join(false)
        }
        (false, true, false, false) => {
            state.write_headers()?;
            state.outer_join(true)
        }
        (false, false, true, false) => {
            state.write_headers()?;
            state.full_outer_join()
        }
        (false, false, false, true) => {
            state.write_headers()?;
            state.cross_join()
        }
        (false, false, false, false) => {
            state.write_headers()?;
            state.inner_join()
        }
        _ => fail!("Please pick exactly one join operation.")
    }
}

struct IoState<R, W: io::Write> {
    wtr: csv::Writer<W>,
    rdr1: csv::Reader<R>,
    sel1: Selection,
    rdr2: csv::Reader<R>,
    sel2: Selection,
    no_headers: bool,
    casei: bool,
    nulls: bool,
}

impl<R: io::Read + io::Seek, W: io::Write> IoState<R, W> {
    fn write_headers(&mut self) -> CliResult<()> {
        if !self.no_headers {
            let mut headers = self.rdr1.byte_headers()?.clone();
            headers.extend(self.rdr2.byte_headers()?.iter());
            self.wtr.write_record(&headers)?;
        }
        Ok(())
    }

    fn inner_join(mut self) -> CliResult<()> {
        let mut scratch = csv::ByteRecord::new();
        let mut validx = ValueIndex::new(
            self.rdr2, &self.sel2, self.casei, self.nulls)?;
        for row in self.rdr1.byte_records() {
            let row = row?;
            let key = get_row_key(&self.sel1, &row, self.casei);
            match validx.values.get(&key) {
                None => continue,
                Some(rows) => {
                    for &rowi in rows.iter() {
                        validx.idx.seek(rowi as u64)?;

                        validx.idx.read_byte_record(&mut scratch)?;
                        let combined = row.iter().chain(scratch.iter());
                        self.wtr.write_record(combined)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn outer_join(mut self, right: bool) -> CliResult<()> {
        if right {
            ::std::mem::swap(&mut self.rdr1, &mut self.rdr2);
            ::std::mem::swap(&mut self.sel1, &mut self.sel2);
        }

        let mut scratch = csv::ByteRecord::new();
        let (_, pad2) = self.get_padding()?;
        let mut validx = ValueIndex::new(
            self.rdr2, &self.sel2, self.casei, self.nulls)?;
        for row in self.rdr1.byte_records() {
            let row = row?;
            let key = get_row_key(&self.sel1, &row, self.casei);
            match validx.values.get(&key) {
                None => {
                    if right {
                        self.wtr.write_record(pad2.iter().chain(&row))?;
                    } else {
                        self.wtr.write_record(row.iter().chain(&pad2))?;
                    }
                }
                Some(rows) => {
                    for &rowi in rows.iter() {
                        validx.idx.seek(rowi as u64)?;
                        let row1 = row.iter();
                        validx.idx.read_byte_record(&mut scratch)?;
                        if right {
                            self.wtr.write_record(scratch.iter().chain(row1))?;
                        } else {
                            self.wtr.write_record(row1.chain(&scratch))?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn full_outer_join(mut self) -> CliResult<()> {
        let mut scratch = csv::ByteRecord::new();
        let (pad1, pad2) = self.get_padding()?;
        let mut validx = ValueIndex::new(
            self.rdr2, &self.sel2, self.casei, self.nulls)?;

        // Keep track of which rows we've written from rdr2.
        let mut rdr2_written: Vec<_> =
            repeat(false).take(validx.num_rows).collect();
        for row1 in self.rdr1.byte_records() {
            let row1 = row1?;
            let key = get_row_key(&self.sel1, &row1, self.casei);
            match validx.values.get(&key) {
                None => {
                    self.wtr.write_record(row1.iter().chain(&pad2))?;
                }
                Some(rows) => {
                    for &rowi in rows.iter() {
                        rdr2_written[rowi] = true;

                        validx.idx.seek(rowi as u64)?;
                        validx.idx.read_byte_record(&mut scratch)?;
                        self.wtr.write_record(row1.iter().chain(&scratch))?;
                    }
                }
            }
        }

        // OK, now write any row from rdr2 that didn't get joined with a row
        // from rdr1.
        for (i, &written) in rdr2_written.iter().enumerate() {
            if !written {
                validx.idx.seek(i as u64)?;
                validx.idx.read_byte_record(&mut scratch)?;
                self.wtr.write_record(pad1.iter().chain(&scratch))?;
            }
        }
        Ok(())
    }

    fn cross_join(mut self) -> CliResult<()> {
        let mut pos = csv::Position::new();
        pos.set_byte(0);
        let mut row2 = csv::ByteRecord::new();
        for row1 in self.rdr1.byte_records() {
            let row1 = row1?;
            self.rdr2.seek(pos.clone())?;
            if self.rdr2.has_headers() {
                // Read and skip the header row, since CSV readers disable
                // the header skipping logic after being seeked.
                self.rdr2.read_byte_record(&mut row2)?;
            }
            while self.rdr2.read_byte_record(&mut row2)? {
                self.wtr.write_record(row1.iter().chain(&row2))?;
            }
        }
        Ok(())
    }

    fn get_padding(
        &mut self,
    ) -> CliResult<(csv::ByteRecord, csv::ByteRecord)> {
        let len1 = self.rdr1.byte_headers()?.len();
        let len2 = self.rdr2.byte_headers()?.len();
        Ok((
            repeat(b"").take(len1).collect(),
            repeat(b"").take(len2).collect(),
        ))
    }
}

impl Args {
    fn new_io_state(&self)
        -> CliResult<IoState<fs::File, Box<io::Write+'static>>> {
        let rconf1 = Config::new(&Some(self.arg_input1.clone()))
            .delimiter(self.flag_delimiter)
            .no_headers(self.flag_no_headers)
            .select(self.arg_columns1.clone());
        let rconf2 = Config::new(&Some(self.arg_input2.clone()))
            .delimiter(self.flag_delimiter)
            .no_headers(self.flag_no_headers)
            .select(self.arg_columns2.clone());

        let mut rdr1 = rconf1.reader_file()?;
        let mut rdr2 = rconf2.reader_file()?;
        let (sel1, sel2) = self.get_selections(
            &rconf1, &mut rdr1, &rconf2, &mut rdr2)?;
        Ok(IoState {
            wtr: Config::new(&self.flag_output).writer()?,
            rdr1: rdr1,
            sel1: sel1,
            rdr2: rdr2,
            sel2: sel2,
            no_headers: rconf1.no_headers,
            casei: self.flag_no_case,
            nulls: self.flag_nulls,
        })
    }

    fn get_selections<R: io::Read>(
        &self,
        rconf1: &Config, rdr1: &mut csv::Reader<R>,
        rconf2: &Config, rdr2: &mut csv::Reader<R>,
    ) -> CliResult<(Selection, Selection)> {
        let headers1 = rdr1.byte_headers()?;
        let headers2 = rdr2.byte_headers()?;
        let select1 = rconf1.selection(&*headers1)?;
        let select2 = rconf2.selection(&*headers2)?;
        if select1.len() != select2.len() {
            return fail!(format!(
                "Column selections must have the same number of columns, \
                 but found column selections with {} and {} columns.",
                select1.len(), select2.len()));
        }
        Ok((select1, select2))
    }
}

struct ValueIndex<R> {
    // This maps tuples of values to corresponding rows.
    values: HashMap<Vec<ByteString>, Vec<usize>>,
    idx: Indexed<R, io::Cursor<Vec<u8>>>,
    num_rows: usize,
}

impl<R: io::Read + io::Seek> ValueIndex<R> {
    fn new(
        mut rdr: csv::Reader<R>,
        sel: &Selection,
        casei: bool,
        nulls: bool,
    ) -> CliResult<ValueIndex<R>> {
        let mut val_idx = HashMap::with_capacity(10000);
        let mut row_idx = io::Cursor::new(Vec::with_capacity(8 * 10000));
        let (mut rowi, mut count) = (0usize, 0usize);

        // This logic is kind of tricky. Basically, we want to include
        // the header row in the line index (because that's what csv::index
        // does), but we don't want to include header values in the ValueIndex.
        if !rdr.has_headers() {
            // ... so if there are no headers, we seek to the beginning and
            // index everything.
            let mut pos = csv::Position::new();
            pos.set_byte(0);
            rdr.seek(pos)?;
        } else {
            // ... and if there are headers, we make sure that we've parsed
            // them, and write the offset of the header row to the index.
            rdr.byte_headers()?;
            row_idx.write_u64::<BigEndian>(0)?;
            count += 1;
        }

        let mut row = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut row)? {
            // This is a bit hokey. We're doing this manually instead of using
            // the `csv-index` crate directly so that we can create both
            // indexes in one pass.
            row_idx.write_u64::<BigEndian>(row.position().unwrap().byte())?;

            let fields: Vec<_> = sel
                .select(&row)
                .map(|v| transform(v, casei))
                .collect();
            if nulls || !fields.iter().any(|f| f.is_empty()) {
                match val_idx.entry(fields) {
                    Entry::Vacant(v) => {
                        let mut rows = Vec::with_capacity(4);
                        rows.push(rowi);
                        v.insert(rows);
                    }
                    Entry::Occupied(mut v) => {
                        v.get_mut().push(rowi);
                    }
                }
            }
            rowi += 1;
            count += 1;
        }

        row_idx.write_u64::<BigEndian>(count as u64)?;
        let idx = Indexed::open(rdr, io::Cursor::new(row_idx.into_inner()))?;
        Ok(ValueIndex {
            values: val_idx,
            idx: idx,
            num_rows: rowi,
        })
    }
}

impl<R> fmt::Debug for ValueIndex<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Sort the values by order of first appearance.
        let mut kvs = self.values.iter().collect::<Vec<_>>();
        kvs.sort_by(|&(_, v1), &(_, v2)| v1[0].cmp(&v2[0]));
        for (keys, rows) in kvs.into_iter() {
            // This is just for debugging, so assume Unicode for now.
            let keys = keys.iter()
                           .map(|k| String::from_utf8(k.to_vec()).unwrap())
                           .collect::<Vec<_>>();
            writeln!(f, "({}) => {:?}", keys.join(", "), rows)?
        }
        Ok(())
    }
}

fn get_row_key(
    sel: &Selection,
    row: &csv::ByteRecord,
    casei: bool,
) -> Vec<ByteString> {
    sel.select(row).map(|v| transform(&v, casei)).collect()
}

fn transform(bs: &[u8], casei: bool) -> ByteString {
    match str::from_utf8(bs) {
        Err(_) => bs.to_vec(),
        Ok(s) => {
            if !casei {
                s.trim().as_bytes().to_vec()
            } else {
                let norm: String =
                    s.trim().chars()
                     .map(|c| c.to_lowercase().next().unwrap()).collect();
                norm.into_bytes()
            }
        }
    }
}

--- MODULE SEPARATOR ---
pub mod cat;
pub mod count;
pub mod fixlengths;
pub mod flatten;
pub mod fmt;
pub mod frequency;
pub mod headers;
pub mod index;
pub mod input;
pub mod join;
pub mod partition;
pub mod reverse;
pub mod sample;
pub mod search;
pub mod select;
pub mod slice;
pub mod sort;
pub mod split;
pub mod stats;
pub mod table;

--- MODULE SEPARATOR ---
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::fs;
use std::io;
use std::path::Path;

use csv;
use regex::Regex;

use CliResult;
use config::{Config, Delimiter};
use select::SelectColumns;
use util::{self, FilenameTemplate};

static USAGE: &'static str = "
Partitions the given CSV data into chunks based on the value of a column

The files are written to the output directory with filenames based on the
values in the partition column and the `--filename` flag.

Usage:
    xsv partition [options] <column> <outdir> [<input>]
    xsv partition --help

partition options:
    --filename <filename>  A filename template to use when constructing
                           the names of the output files.  The string '{}'
                           will be replaced by a value based on the value
                           of the field, but sanitized for shell safety.
                           [default: {}.csv]
    -p, --prefix-length <n>  Truncate the partition column after the
                           specified number of bytes when creating the
                           output file.
    --drop                 Drop the partition column from results.

Common options:
    -h, --help             Display this message
    -n, --no-headers       When set, the first row will NOT be interpreted
                           as column names. Otherwise, the first row will
                           appear in all chunks as the header row.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Clone, Deserialize)]
struct Args {
    arg_column: SelectColumns,
    arg_input: Option<String>,
    arg_outdir: String,
    flag_filename: FilenameTemplate,
    flag_prefix_length: Option<usize>,
    flag_drop: bool,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    fs::create_dir_all(&args.arg_outdir)?;

    // It would be nice to support efficient parallel partitions, but doing
    // do would involve more complicated inter-thread communication, with
    // multiple readers and writers, and some way of passing buffers
    // between them.
    args.sequential_partition()
}

impl Args {
    /// Configuration for our reader.
    fn rconfig(&self) -> Config {
        Config::new(&self.arg_input)
            .delimiter(self.flag_delimiter)
            .no_headers(self.flag_no_headers)
            .select(self.arg_column.clone())
    }

    /// Get the column to use as a key.
    fn key_column(
        &self,
        rconfig: &Config,
        headers: &csv::ByteRecord,
    ) -> CliResult<usize> {
        let select_cols = rconfig.selection(headers)?;
        if select_cols.len() == 1 {
            Ok(select_cols[0])
        } else {
            fail!("can only partition on one column")
        }
    }

    /// A basic sequential partition.
    fn sequential_partition(&self) -> CliResult<()> {
        let rconfig = self.rconfig();
        let mut rdr = rconfig.reader()?;
        let headers = rdr.byte_headers()?.clone();
        let key_col = self.key_column(&rconfig, &headers)?;
        let mut gen = WriterGenerator::new(self.flag_filename.clone());

        let mut writers: HashMap<Vec<u8>, BoxedWriter> =
            HashMap::new();
        let mut row = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut row)? {
            // Decide what file to put this in.
            let column = &row[key_col];
            let key = match self.flag_prefix_length {
                // We exceed --prefix-length, so ignore the extra bytes.
                Some(len) if len < column.len() => &column[0..len],
                _ => &column[..],
            };
            let mut entry = writers.entry(key.to_vec());
            let wtr = match entry {
                Entry::Occupied(ref mut occupied) => occupied.get_mut(),
                Entry::Vacant(vacant) => {
                    // We have a new key, so make a new writer.
                    let mut wtr = gen.writer(&*self.arg_outdir, key)?;
                    if !rconfig.no_headers {
                        if self.flag_drop {
                            wtr.write_record(headers.iter().enumerate()
                                .filter_map(|(i, e)| if i != key_col { Some(e) } else { None }))?;
                        } else {
                            wtr.write_record(&headers)?;
                        }
                    }
                    vacant.insert(wtr)
                }
            };
            if self.flag_drop {
                wtr.write_record(row.iter().enumerate()
                    .filter_map(|(i, e)| if i != key_col { Some(e) } else { None }))?;
            } else {
                wtr.write_byte_record(&row)?;
            }
        }
        Ok(())
    }
}

type BoxedWriter = csv::Writer<Box<io::Write+'static>>;

/// Generates unique filenames based on CSV values.
struct WriterGenerator {
    template: FilenameTemplate,
    counter: usize,
    used: HashSet<String>,
    non_word_char: Regex,
}

impl WriterGenerator {
    fn new(template: FilenameTemplate) -> WriterGenerator {
        WriterGenerator {
            template: template,
            counter: 1,
            used: HashSet::new(),
            non_word_char: Regex::new(r"\W").unwrap(),
        }
    }

    /// Create a CSV writer for `key`.  Does not add headers.
    fn writer<P>(&mut self, path: P, key: &[u8]) -> io::Result<BoxedWriter>
        where P: AsRef<Path>
    {
        let unique_value = self.unique_value(key);
        self.template.writer(path.as_ref(), &unique_value)
    }

    /// Generate a unique value for `key`, suitable for use in a
    /// "shell-safe" filename.  If you pass `key` twice, you'll get two
    /// different values.
    fn unique_value(&mut self, key: &[u8]) -> String {
        // Sanitize our key.
        let utf8 = String::from_utf8_lossy(key);
        let safe = self.non_word_char.replace_all(&*utf8, "").into_owned();
        let base =
            if safe.is_empty() {
                "empty".to_owned()
            } else {
                safe
            };

        // Now check for collisions.
        if !self.used.contains(&base) {
            self.used.insert(base.clone());
            base
        } else {
            loop {
                let candidate = format!("{}_{}", &base, self.counter);
                self.counter = self.counter.checked_add(1).unwrap_or_else(|| {
                    // We'll run out of other things long before we ever
                    // reach this, but we'll check just for correctness and
                    // completeness.
                    panic!("Cannot generate unique value")
                });
                if !self.used.contains(&candidate) {
                    self.used.insert(candidate.clone());
                    return candidate
                }
            }
        }
    }
}

--- MODULE SEPARATOR ---
use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Reverses rows of CSV data.

Useful for cases when there is no column that can be used for sorting in reverse order,
or when keys are not unique and order of rows with the same key needs to be preserved.

Note that this requires reading all of the CSV data into memory.

Usage:
    xsv reverse [options] [<input>]

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will not be interpreted
                           as headers. Namely, it will be reversed with the rest
                           of the rows. Otherwise, the first row will always
                           appear as the header row in the output.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(args.flag_no_headers);

    let mut rdr = rconfig.reader()?;

    let mut all = rdr.byte_records().collect::<Result<Vec<_>, _>>()?;
    all.reverse();

    let mut wtr = Config::new(&args.flag_output).writer()?;
    rconfig.write_headers(&mut rdr, &mut wtr)?;
    for r in all.into_iter() {
        wtr.write_byte_record(&r)?;
    }
    Ok(wtr.flush()?)
}

--- MODULE SEPARATOR ---
use std::io;

use byteorder::{ByteOrder, LittleEndian};
use csv;
use rand::{self, Rng, SeedableRng};
use rand::rngs::StdRng;

use CliResult;
use config::{Config, Delimiter};
use index::Indexed;
use util;

static USAGE: &'static str = "
Randomly samples CSV data uniformly using memory proportional to the size of
the sample.

When an index is present, this command will use random indexing if the sample
size is less than 10% of the total number of records. This allows for efficient
sampling such that the entire CSV file is not parsed.

This command is intended to provide a means to sample from a CSV data set that
is too big to fit into memory (for example, for use with commands like 'xsv
frequency' or 'xsv stats'). It will however visit every CSV record exactly
once, which is necessary to provide a uniform random sample. If you wish to
limit the number of records visited, use the 'xsv slice' command to pipe into
'xsv sample'.

Usage:
    xsv sample [options] <sample-size> [<input>]
    xsv sample --help

sample options:
    --seed <number>        RNG seed.

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will be consider as part of
                           the population to sample from. (When not set, the
                           first row is the header row and will always appear
                           in the output.)
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    arg_sample_size: u64,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
    flag_seed: Option<usize>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(args.flag_no_headers);
    let sample_size = args.arg_sample_size;

    let mut wtr = Config::new(&args.flag_output).writer()?;
    let sampled = match rconfig.indexed()? {
        Some(mut idx) => {
            if do_random_access(sample_size, idx.count()) {
                rconfig.write_headers(&mut *idx, &mut wtr)?;
                sample_random_access(&mut idx, sample_size)?
            } else {
                let mut rdr = rconfig.reader()?;
                rconfig.write_headers(&mut rdr, &mut wtr)?;
                sample_reservoir(&mut rdr, sample_size, args.flag_seed)?
            }
        }
        _ => {
            let mut rdr = rconfig.reader()?;
            rconfig.write_headers(&mut rdr, &mut wtr)?;
            sample_reservoir(&mut rdr, sample_size, args.flag_seed)?
        }
    };
    for row in sampled.into_iter() {
        wtr.write_byte_record(&row)?;
    }
    Ok(wtr.flush()?)
}

fn sample_random_access<R, I>(
    idx: &mut Indexed<R, I>,
    sample_size: u64,
) -> CliResult<Vec<csv::ByteRecord>>
where R: io::Read + io::Seek, I: io::Read + io::Seek
{
    let mut all_indices = (0..idx.count()).collect::<Vec<_>>();
    let mut rng = ::rand::thread_rng();
    rng.shuffle(&mut *all_indices);

    let mut sampled = Vec::with_capacity(sample_size as usize);
    for i in all_indices.into_iter().take(sample_size as usize) {
        idx.seek(i)?;
        sampled.push(idx.byte_records().next().unwrap()?);
    }
    Ok(sampled)
}

fn sample_reservoir<R: io::Read>(
    rdr: &mut csv::Reader<R>,
    sample_size: u64,
    seed: Option<usize>
) -> CliResult<Vec<csv::ByteRecord>> {
    // The following algorithm has been adapted from:
    // https://en.wikipedia.org/wiki/Reservoir_sampling
    let mut reservoir = Vec::with_capacity(sample_size as usize);
    let mut records = rdr.byte_records().enumerate();
    for (_, row) in records.by_ref().take(reservoir.capacity()) {
        reservoir.push(row?);
    }

    // Seeding rng
    let mut rng: StdRng = match seed {
        None => {
            StdRng::from_rng(rand::thread_rng()).unwrap()
        }
        Some(seed) => {
            let mut buf = [0u8; 32];
            LittleEndian::write_u64(&mut buf, seed as u64);
            SeedableRng::from_seed(buf)
        }
    };

    // Now do the sampling.
    for (i, row) in records {
        let random = rng.gen_range(0, i+1);
        if random < sample_size as usize {
            reservoir[random] = row?;
        }
    }
    Ok(reservoir)
}

fn do_random_access(sample_size: u64, total: u64) -> bool {
    sample_size <= (total / 10)
}

--- MODULE SEPARATOR ---
use csv;
use regex::bytes::RegexBuilder;

use CliResult;
use config::{Config, Delimiter};
use select::SelectColumns;
use util;

static USAGE: &'static str = "
Filters CSV data by whether the given regex matches a row.

The regex is applied to each field in each row, and if any field matches,
then the row is written to the output. The columns to search can be limited
with the '--select' flag (but the full row is still written to the output if
there is a match).

Usage:
    xsv search [options] <regex> [<input>]
    xsv search --help

search options:
    -i, --ignore-case      Case insensitive search. This is equivalent to
                           prefixing the regex with '(?i)'.
    -s, --select <arg>     Select the columns to search. See 'xsv select -h'
                           for the full syntax.
    -v, --invert-match     Select only rows that did not match

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will not be interpreted
                           as headers. (i.e., They are not searched, analyzed,
                           sliced, etc.)
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    arg_regex: String,
    flag_select: SelectColumns,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
    flag_invert_match: bool,
    flag_ignore_case: bool,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let pattern = RegexBuilder::new(&*args.arg_regex)
        .case_insensitive(args.flag_ignore_case)
        .build()?;
    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(args.flag_no_headers)
        .select(args.flag_select);

    let mut rdr = rconfig.reader()?;
    let mut wtr = Config::new(&args.flag_output).writer()?;

    let headers = rdr.byte_headers()?.clone();
    let sel = rconfig.selection(&headers)?;

    if !rconfig.no_headers {
        wtr.write_record(&headers)?;
    }
    let mut record = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut record)? {
        let mut m = sel.select(&record).any(|f| pattern.is_match(f));
        if args.flag_invert_match {
            m = !m;
        }
        if m {
            wtr.write_byte_record(&record)?;
        }
    }
    Ok(wtr.flush()?)
}

--- MODULE SEPARATOR ---
use csv;

use CliResult;
use config::{Config, Delimiter};
use select::SelectColumns;
use util;

static USAGE: &'static str = "
Select columns from CSV data efficiently.

This command lets you manipulate the columns in CSV data. You can re-order
them, duplicate them or drop them. Columns can be referenced by index or by
name if there is a header row (duplicate column names can be disambiguated with
more indexing). Finally, column ranges can be specified.

  Select the first and fourth columns:
  $ xsv select 1,4

  Select the first 4 columns (by index and by name):
  $ xsv select 1-4
  $ xsv select Header1-Header4

  Ignore the first 2 columns (by range and by omission):
  $ xsv select 3-
  $ xsv select '!1-2'

  Select the third column named 'Foo':
  $ xsv select 'Foo[2]'

  Re-order and duplicate columns arbitrarily:
  $ xsv select 3-1,Header3-Header1,Header1,Foo[2],Header1

  Quote column names that conflict with selector syntax:
  $ xsv select '\"Date - Opening\",\"Date - Actual Closing\"'

Usage:
    xsv select [options] [--] <selection> [<input>]
    xsv select --help

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will not be interpreted
                           as headers. (i.e., They are not searched, analyzed,
                           sliced, etc.)
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    arg_selection: SelectColumns,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;

    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(args.flag_no_headers)
        .select(args.arg_selection);

    let mut rdr = rconfig.reader()?;
    let mut wtr = Config::new(&args.flag_output).writer()?;

    let headers = rdr.byte_headers()?.clone();
    let sel = rconfig.selection(&headers)?;

    if !rconfig.no_headers {
        wtr.write_record(sel.iter().map(|&i| &headers[i]))?;
    }
    let mut record = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut record)? {
        wtr.write_record(sel.iter().map(|&i| &record[i]))?;
    }
    wtr.flush()?;
    Ok(())
}

--- MODULE SEPARATOR ---
use std::fs;


use CliResult;
use config::{Config, Delimiter};
use index::Indexed;
use util;

static USAGE: &'static str = "
Returns the rows in the range specified (starting at 0, half-open interval).
The range does not include headers.

If the start of the range isn't specified, then the slice starts from the first
record in the CSV data.

If the end of the range isn't specified, then the slice continues to the last
record in the CSV data.

This operation can be made much faster by creating an index with 'xsv index'
first. Namely, a slice on an index requires parsing just the rows that are
sliced. Without an index, all rows up to the first row in the slice must be
parsed.

Usage:
    xsv slice [options] [<input>]

slice options:
    -s, --start <arg>      The index of the record to slice from.
    -e, --end <arg>        The index of the record to slice to.
    -l, --len <arg>        The length of the slice (can be used instead
                           of --end).
    -i, --index <arg>      Slice a single record (shortcut for -s N -l 1).

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will not be interpreted
                           as headers. Otherwise, the first row will always
                           appear in the output as the header row.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_start: Option<usize>,
    flag_end: Option<usize>,
    flag_len: Option<usize>,
    flag_index: Option<usize>,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    match args.rconfig().indexed()? {
        None => args.no_index(),
        Some(idxed) => args.with_index(idxed),
    }
}

impl Args {
    fn no_index(&self) -> CliResult<()> {
        let mut rdr = self.rconfig().reader()?;
        let mut wtr = self.wconfig().writer()?;
        self.rconfig().write_headers(&mut rdr, &mut wtr)?;

        let (start, end) = self.range()?;
        for r in rdr.byte_records().skip(start).take(end - start) {
            wtr.write_byte_record(&r?)?;
        }
        Ok(wtr.flush()?)
    }

    fn with_index(
        &self,
        mut idx: Indexed<fs::File, fs::File>,
    ) -> CliResult<()> {
        let mut wtr = self.wconfig().writer()?;
        self.rconfig().write_headers(&mut *idx, &mut wtr)?;

        let (start, end) = self.range()?;
        if end - start == 0 {
            return Ok(());
        }
        idx.seek(start as u64)?;
        for r in idx.byte_records().take(end - start) {
            wtr.write_byte_record(&r?)?;
        }
        wtr.flush()?;
        Ok(())
    }

    fn range(&self) -> Result<(usize, usize), String> {
        util::range(
            self.flag_start, self.flag_end, self.flag_len, self.flag_index)
    }

    fn rconfig(&self) -> Config {
        Config::new(&self.arg_input)
            .delimiter(self.flag_delimiter)
            .no_headers(self.flag_no_headers)
    }

    fn wconfig(&self) -> Config {
        Config::new(&self.flag_output)
    }
}

--- MODULE SEPARATOR ---
use std::cmp;

use CliResult;
use config::{Config, Delimiter};
use select::SelectColumns;
use util;
use std::str::from_utf8;

use self::Number::{Float, Int};

static USAGE: &'static str = "
Sorts CSV data lexicographically.

Note that this requires reading all of the CSV data into memory.

Usage:
    xsv sort [options] [<input>]

sort options:
    -s, --select <arg>     Select a subset of columns to sort.
                           See 'xsv select --help' for the format details.
    -N, --numeric          Compare according to string numerical value
    -R, --reverse          Reverse order

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will not be interpreted
                           as headers. Namely, it will be sorted with the rest
                           of the rows. Otherwise, the first row will always
                           appear as the header row in the output.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_select: SelectColumns,
    flag_numeric: bool,
    flag_reverse: bool,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let numeric = args.flag_numeric;
    let reverse = args.flag_reverse;
    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(args.flag_no_headers)
        .select(args.flag_select);

    let mut rdr = rconfig.reader()?;

    let headers = rdr.byte_headers()?.clone();
    let sel = rconfig.selection(&headers)?;

    let mut all = rdr.byte_records().collect::<Result<Vec<_>, _>>()?;
    match (numeric, reverse) {
        (false, false) =>
            all.sort_by(|r1, r2| {
                let a = sel.select(r1);
                let b = sel.select(r2);
                iter_cmp(a, b)
            }),
        (true, false) =>
            all.sort_by(|r1, r2| {
                let a = sel.select(r1);
                let b = sel.select(r2);
                iter_cmp_num(a, b)
            }),
        (false, true) =>
            all.sort_by(|r1, r2| {
                let a = sel.select(r1);
                let b = sel.select(r2);
                iter_cmp(b, a)
            }),
        (true, true) =>
            all.sort_by(|r1, r2| {
                let a = sel.select(r1);
                let b = sel.select(r2);
                iter_cmp_num(b, a)
            }),
    }

    let mut wtr = Config::new(&args.flag_output).writer()?;
    rconfig.write_headers(&mut rdr, &mut wtr)?;
    for r in all.into_iter() {
        wtr.write_byte_record(&r)?;
    }
    Ok(wtr.flush()?)
}

/// Order `a` and `b` lexicographically using `Ord`
pub fn iter_cmp<A, L, R>(mut a: L, mut b: R) -> cmp::Ordering
        where A: Ord, L: Iterator<Item=A>, R: Iterator<Item=A> {
    loop {
        match (a.next(), b.next()) {
            (None, None) => return cmp::Ordering::Equal,
            (None, _   ) => return cmp::Ordering::Less,
            (_   , None) => return cmp::Ordering::Greater,
            (Some(x), Some(y)) => match x.cmp(&y) {
                cmp::Ordering::Equal => (),
                non_eq => return non_eq,
            },
        }
    }
}

/// Try parsing `a` and `b` as numbers when ordering
pub fn iter_cmp_num<'a, L, R>(mut a: L, mut b: R) -> cmp::Ordering
        where L: Iterator<Item=&'a [u8]>, R: Iterator<Item=&'a [u8]> {
    loop {
        match (next_num(&mut a), next_num(&mut b)) {
            (None, None) => return cmp::Ordering::Equal,
            (None, _   ) => return cmp::Ordering::Less,
            (_   , None) => return cmp::Ordering::Greater,
            (Some(x), Some(y)) => match compare_num(x, y) {
                cmp::Ordering::Equal => (),
                non_eq => return non_eq,
            },
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Number {
    Int(i64),
    Float(f64),
}

fn compare_num(n1: Number, n2: Number) -> cmp::Ordering{
    match (n1, n2) {
        (Int(i1), Int(i2)) => i1.cmp(&i2),
        (Int(i1), Float(f2)) => compare_float(i1 as f64, f2),
        (Float(f1), Int(i2)) => compare_float(f1, i2 as f64),
        (Float(f1), Float(f2)) => compare_float(f1, f2),
    }
}

fn compare_float(f1: f64, f2: f64) -> cmp::Ordering {
    f1.partial_cmp(&f2).unwrap_or(cmp::Ordering::Equal)
}



fn next_num<'a, X>(xs: &mut X) -> Option<Number>
        where X: Iterator<Item=&'a [u8]> {
    xs.next()
        .and_then(|bytes| from_utf8(bytes).ok())
        .and_then(|s| {
            if let Ok(i) = s.parse::<i64>() { Some(Number::Int(i)) }
            else if let Ok(f) = s.parse::<f64>() { Some(Number::Float(f)) }
            else { None }
        })
}

--- MODULE SEPARATOR ---
use std::fs;
use std::io;
use std::path::Path;

use channel;
use csv;
use threadpool::ThreadPool;

use CliResult;
use config::{Config, Delimiter};
use index::Indexed;
use util::{self, FilenameTemplate};

static USAGE: &'static str = "
Splits the given CSV data into chunks.

The files are written to the directory given with the name '{start}.csv',
where {start} is the index of the first record of the chunk (starting at 0).

Usage:
    xsv split [options] <outdir> [<input>]
    xsv split --help

split options:
    -s, --size <arg>       The number of records to write into each chunk.
                           [default: 500]
    -j, --jobs <arg>       The number of spliting jobs to run in parallel.
                           This only works when the given CSV data has
                           an index already created. Note that a file handle
                           is opened for each job.
                           When set to '0', the number of jobs is set to the
                           number of CPUs detected.
                           [default: 0]
    --filename <filename>  A filename template to use when constructing
                           the names of the output files.  The string '{}'
                           will be replaced by a value based on the value
                           of the field, but sanitized for shell safety.
                           [default: {}.csv]

Common options:
    -h, --help             Display this message
    -n, --no-headers       When set, the first row will NOT be interpreted
                           as column names. Otherwise, the first row will
                           appear in all chunks as the header row.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Clone, Deserialize)]
struct Args {
    arg_input: Option<String>,
    arg_outdir: String,
    flag_size: usize,
    flag_jobs: usize,
    flag_filename: FilenameTemplate,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    if args.flag_size == 0 {
        return fail!("--size must be greater than 0.");
    }
    fs::create_dir_all(&args.arg_outdir)?;

    match args.rconfig().indexed()? {
        Some(idx) => args.parallel_split(idx),
        None => args.sequential_split(),
    }
}

impl Args {
    fn sequential_split(&self) -> CliResult<()> {
        let rconfig = self.rconfig();
        let mut rdr = rconfig.reader()?;
        let headers = rdr.byte_headers()?.clone();

        let mut wtr = self.new_writer(&headers, 0)?;
        let mut i = 0;
        let mut row = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut row)? {
            if i > 0 && i % self.flag_size == 0 {
                wtr.flush()?;
                wtr = self.new_writer(&headers, i)?;
            }
            wtr.write_byte_record(&row)?;
            i += 1;
        }
        wtr.flush()?;
        Ok(())
    }

    fn parallel_split(
        &self,
        idx: Indexed<fs::File, fs::File>,
    ) -> CliResult<()> {
        let nchunks = util::num_of_chunks(
            idx.count() as usize, self.flag_size);
        let pool = ThreadPool::new(self.njobs());
        let (tx, rx) = channel::bounded::<()>(0);
        for i in 0..nchunks {
            let args = self.clone();
            let tx = tx.clone();
            pool.execute(move || {
                let conf = args.rconfig();
                let mut idx = conf.indexed().unwrap().unwrap();
                let headers = idx.byte_headers().unwrap().clone();
                let mut wtr = args
                    .new_writer(&headers, i * args.flag_size)
                    .unwrap();

                idx.seek((i * args.flag_size) as u64).unwrap();
                for row in idx.byte_records().take(args.flag_size) {
                    let row = row.unwrap();
                    wtr.write_byte_record(&row).unwrap();
                }
                wtr.flush().unwrap();
                drop(tx);
            });
        }
        drop(tx);
        rx.recv();
        Ok(())
    }

    fn new_writer(
        &self,
        headers: &csv::ByteRecord,
        start: usize,
    ) -> CliResult<csv::Writer<Box<io::Write+'static>>> {
        let dir = Path::new(&self.arg_outdir);
        let path = dir.join(self.flag_filename.filename(&format!("{}", start)));
        let spath = Some(path.display().to_string());
        let mut wtr = Config::new(&spath).writer()?;
        if !self.rconfig().no_headers {
            wtr.write_record(headers)?;
        }
        Ok(wtr)
    }

    fn rconfig(&self) -> Config {
        Config::new(&self.arg_input)
            .delimiter(self.flag_delimiter)
            .no_headers(self.flag_no_headers)
    }

    fn njobs(&self) -> usize {
        if self.flag_jobs == 0 {
            util::num_cpus()
        } else {
            self.flag_jobs
        }
    }
}

--- MODULE SEPARATOR ---
use std::borrow::ToOwned;
use std::default::Default;
use std::fmt;
use std::fs;
use std::io;
use std::iter::{FromIterator, repeat};
use std::str::{self, FromStr};

use channel;
use csv;
use stats::{Commute, OnlineStats, MinMax, Unsorted, merge_all};
use threadpool::ThreadPool;

use CliResult;
use config::{Config, Delimiter};
use index::Indexed;
use select::{SelectColumns, Selection};
use util;

use self::FieldType::{TUnknown, TNull, TUnicode, TFloat, TInteger};

static USAGE: &'static str = "
Computes basic statistics on CSV data.

Basic statistics includes mean, median, mode, standard deviation, sum, max and
min values. Note that some statistics are expensive to compute, so they must
be enabled explicitly. By default, the following statistics are reported for
*every* column in the CSV data: mean, max, min and standard deviation. The
default set of statistics corresponds to statistics that can be computed
efficiently on a stream of data (i.e., constant memory).

Computing statistics on a large file can be made much faster if you create
an index for it first with 'xsv index'.

Usage:
    xsv stats [options] [<input>]

stats options:
    -s, --select <arg>     Select a subset of columns to compute stats for.
                           See 'xsv select --help' for the format details.
                           This is provided here because piping 'xsv select'
                           into 'xsv stats' will disable the use of indexing.
    --everything           Show all statistics available.
    --mode                 Show the mode.
                           This requires storing all CSV data in memory.
    --cardinality          Show the cardinality.
                           This requires storing all CSV data in memory.
    --median               Show the median.
                           This requires storing all CSV data in memory.
    --nulls                Include NULLs in the population size for computing
                           mean and standard deviation.
    -j, --jobs <arg>       The number of jobs to run in parallel.
                           This works better when the given CSV data has
                           an index already created. Note that a file handle
                           is opened for each job.
                           When set to '0', the number of jobs is set to the
                           number of CPUs detected.
                           [default: 0]

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -n, --no-headers       When set, the first row will NOT be interpreted
                           as column names. i.e., They will be included
                           in statistics.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Clone, Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_select: SelectColumns,
    flag_everything: bool,
    flag_mode: bool,
    flag_cardinality: bool,
    flag_median: bool,
    flag_nulls: bool,
    flag_jobs: usize,
    flag_output: Option<String>,
    flag_no_headers: bool,
    flag_delimiter: Option<Delimiter>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;

    let mut wtr = Config::new(&args.flag_output).writer()?;
    let (headers, stats) = match args.rconfig().indexed()? {
        None => args.sequential_stats(),
        Some(idx) => {
            if args.flag_jobs == 1 {
                args.sequential_stats()
            } else {
                args.parallel_stats(idx)
            }
        }
    }?;
    let stats = args.stats_to_records(stats);

    wtr.write_record(&args.stat_headers())?;
    let fields = headers.iter().zip(stats.into_iter());
    for (i, (header, stat)) in fields.enumerate() {
        let header =
            if args.flag_no_headers {
                i.to_string().into_bytes()
            } else {
                header.to_vec()
            };
        let stat = stat.iter().map(|f| f.as_bytes());
        wtr.write_record(vec![&*header].into_iter().chain(stat))?;
    }
    wtr.flush()?;
    Ok(())
}

impl Args {
    fn sequential_stats(&self) -> CliResult<(csv::ByteRecord, Vec<Stats>)> {
        let mut rdr = self.rconfig().reader()?;
        let (headers, sel) = self.sel_headers(&mut rdr)?;
        let stats = self.compute(&sel, rdr.byte_records())?;
        Ok((headers, stats))
    }

    fn parallel_stats(
        &self,
        idx: Indexed<fs::File, fs::File>,
    ) -> CliResult<(csv::ByteRecord, Vec<Stats>)> {
        // N.B. This method doesn't handle the case when the number of records
        // is zero correctly. So we use `sequential_stats` instead.
        if idx.count() == 0 {
            return self.sequential_stats();
        }

        let mut rdr = self.rconfig().reader()?;
        let (headers, sel) = self.sel_headers(&mut rdr)?;

        let chunk_size = util::chunk_size(idx.count() as usize, self.njobs());
        let nchunks = util::num_of_chunks(idx.count() as usize, chunk_size);

        let pool = ThreadPool::new(self.njobs());
        let (send, recv) = channel::bounded(0);
        for i in 0..nchunks {
            let (send, args, sel) = (send.clone(), self.clone(), sel.clone());
            pool.execute(move || {
                let mut idx = args.rconfig().indexed().unwrap().unwrap();
                idx.seek((i * chunk_size) as u64).unwrap();
                let it = idx.byte_records().take(chunk_size);
                send.send(args.compute(&sel, it).unwrap());
            });
        }
        drop(send);
        Ok((headers, merge_all(recv).unwrap_or_else(Vec::new)))
    }

    fn stats_to_records(&self, stats: Vec<Stats>) -> Vec<csv::StringRecord> {
        let mut records: Vec<_> = repeat(csv::StringRecord::new())
            .take(stats.len())
            .collect();
        let pool = ThreadPool::new(self.njobs());
        let mut results = vec![];
        for mut stat in stats.into_iter() {
            let (send, recv) = channel::bounded(0);
            results.push(recv);
            pool.execute(move || { send.send(stat.to_record()); });
        }
        for (i, recv) in results.into_iter().enumerate() {
            records[i] = recv.recv().unwrap();
        }
        records
    }

    fn compute<I>(&self, sel: &Selection, it: I) -> CliResult<Vec<Stats>>
            where I: Iterator<Item=csv::Result<csv::ByteRecord>> {
        let mut stats = self.new_stats(sel.len());
        for row in it {
            let row = row?;
            for (i, field) in sel.select(&row).enumerate() {
                stats[i].add(field);
            }
        }
        Ok(stats)
    }

    fn sel_headers<R: io::Read>(
        &self,
        rdr: &mut csv::Reader<R>,
    ) -> CliResult<(csv::ByteRecord, Selection)> {
        let headers = rdr.byte_headers()?.clone();
        let sel = self.rconfig().selection(&headers)?;
        Ok((csv::ByteRecord::from_iter(sel.select(&headers)), sel))
    }

    fn rconfig(&self) -> Config {
        Config::new(&self.arg_input)
            .delimiter(self.flag_delimiter)
            .no_headers(self.flag_no_headers)
            .select(self.flag_select.clone())
    }

    fn njobs(&self) -> usize {
        if self.flag_jobs == 0 { util::num_cpus() } else { self.flag_jobs }
    }

    fn new_stats(&self, record_len: usize) -> Vec<Stats> {
        repeat(Stats::new(WhichStats {
            include_nulls: self.flag_nulls,
            sum: true,
            range: true,
            dist: true,
            cardinality: self.flag_cardinality || self.flag_everything,
            median: self.flag_median || self.flag_everything,
            mode: self.flag_mode || self.flag_everything,
        })).take(record_len).collect()
    }

    fn stat_headers(&self) -> csv::StringRecord {
        let mut fields = vec![
            "field", "type", "sum", "min", "max", "min_length", "max_length",
            "mean", "stddev",
        ];
        let all = self.flag_everything;
        if self.flag_median || all { fields.push("median"); }
        if self.flag_mode || all { fields.push("mode"); }
        if self.flag_cardinality || all { fields.push("cardinality"); }
        csv::StringRecord::from(fields)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WhichStats {
    include_nulls: bool,
    sum: bool,
    range: bool,
    dist: bool,
    cardinality: bool,
    median: bool,
    mode: bool,
}

impl Commute for WhichStats {
    fn merge(&mut self, other: WhichStats) {
        assert_eq!(*self, other);
    }
}

#[derive(Clone)]
struct Stats {
    typ: FieldType,
    sum: Option<TypedSum>,
    minmax: Option<TypedMinMax>,
    online: Option<OnlineStats>,
    mode: Option<Unsorted<Vec<u8>>>,
    median: Option<Unsorted<f64>>,
    which: WhichStats,
}

impl Stats {
    fn new(which: WhichStats) -> Stats {
        let (mut sum, mut minmax, mut online, mut mode, mut median) =
            (None, None, None, None, None);
        if which.sum { sum = Some(Default::default()); }
        if which.range { minmax = Some(Default::default()); }
        if which.dist { online = Some(Default::default()); }
        if which.mode || which.cardinality { mode = Some(Default::default()); }
        if which.median { median = Some(Default::default()); }
        Stats {
            typ: Default::default(),
            sum: sum,
            minmax: minmax,
            online: online,
            mode: mode,
            median: median,
            which: which,
        }
    }

    fn add(&mut self, sample: &[u8]) {
        let sample_type = FieldType::from_sample(sample);
        self.typ.merge(sample_type);

        let t = self.typ;
        self.sum.as_mut().map(|v| v.add(t, sample));
        self.minmax.as_mut().map(|v| v.add(t, sample));
        self.mode.as_mut().map(|v| v.add(sample.to_vec()));
        match self.typ {
            TUnknown => {}
            TNull => {
                if self.which.include_nulls {
                    self.online.as_mut().map(|v| { v.add_null(); });
                }
            }
            TUnicode => {}
            TFloat | TInteger => {
                if sample_type.is_null() {
                    if self.which.include_nulls {
                        self.online.as_mut().map(|v| { v.add_null(); });
                    }
                } else {
                    let n = from_bytes::<f64>(sample).unwrap();
                    self.median.as_mut().map(|v| { v.add(n); });
                    self.online.as_mut().map(|v| { v.add(n); });
                }
            }
        }
    }

    fn to_record(&mut self) -> csv::StringRecord {
        let typ = self.typ;
        let mut pieces = vec![];
        let empty = || "".to_owned();

        pieces.push(self.typ.to_string());
        match self.sum.as_ref().and_then(|sum| sum.show(typ)) {
            Some(sum) => { pieces.push(sum); }
            None => { pieces.push(empty()); }
        }
        match self.minmax.as_ref().and_then(|mm| mm.show(typ)) {
            Some(mm) => { pieces.push(mm.0); pieces.push(mm.1); }
            None => { pieces.push(empty()); pieces.push(empty()); }
        }
        match self.minmax.as_ref().and_then(|mm| mm.len_range()) {
            Some(mm) => { pieces.push(mm.0); pieces.push(mm.1); }
            None => { pieces.push(empty()); pieces.push(empty()); }
        }

        if !self.typ.is_number() {
            pieces.push(empty()); pieces.push(empty());
        } else {
            match self.online {
                Some(ref v) => {
                    pieces.push(v.mean().to_string());
                    pieces.push(v.stddev().to_string());
                }
                None => { pieces.push(empty()); pieces.push(empty()); }
            }
        }
        match self.median.as_mut().and_then(|v| v.median()) {
            None => {
                if self.which.median {
                    pieces.push(empty());
                }
            }
            Some(v) => { pieces.push(v.to_string()); }
        }
        match self.mode.as_mut() {
            None => {
                if self.which.mode {
                    pieces.push(empty());
                }
                if self.which.cardinality {
                    pieces.push(empty());
                }
            }
            Some(ref mut v) => {
                if self.which.mode {
                    let lossy = |s: Vec<u8>| -> String {
                        String::from_utf8_lossy(&*s).into_owned()
                    };
                    pieces.push(
                        v.mode().map_or("N/A".to_owned(), lossy));
                }
                if self.which.cardinality {
                    pieces.push(v.cardinality().to_string());
                }
            }
        }
        csv::StringRecord::from(pieces)
    }
}

impl Commute for Stats {
    fn merge(&mut self, other: Stats) {
        self.typ.merge(other.typ);
        self.sum.merge(other.sum);
        self.minmax.merge(other.minmax);
        self.online.merge(other.online);
        self.mode.merge(other.mode);
        self.median.merge(other.median);
        self.which.merge(other.which);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum FieldType {
    TUnknown,
    TNull,
    TUnicode,
    TFloat,
    TInteger,
}

impl FieldType {
    fn from_sample(sample: &[u8]) -> FieldType {
        if sample.is_empty() {
            return TNull;
        }
        let string = match str::from_utf8(sample) {
            Err(_) => return TUnknown,
            Ok(s) => s,
        };
        if let Ok(_) = string.parse::<i64>() { return TInteger; }
        if let Ok(_) = string.parse::<f64>() { return TFloat; }
        TUnicode
    }

    fn is_number(&self) -> bool {
        *self == TFloat || *self == TInteger
    }

    fn is_null(&self) -> bool {
        *self == TNull
    }
}

impl Commute for FieldType {
    fn merge(&mut self, other: FieldType) {
        *self = match (*self, other) {
            (TUnicode, TUnicode) => TUnicode,
            (TFloat, TFloat) => TFloat,
            (TInteger, TInteger) => TInteger,
            // Null does not impact the type.
            (TNull, any) | (any, TNull) => any,
            // There's no way to get around an unknown.
            (TUnknown, _) | (_, TUnknown) => TUnknown,
            // Integers can degrate to floats.
            (TFloat, TInteger) | (TInteger, TFloat) => TFloat,
            // Numbers can degrade to Unicode strings.
            (TUnicode, TFloat) | (TFloat, TUnicode) => TUnicode,
            (TUnicode, TInteger) | (TInteger, TUnicode) => TUnicode,
        };
    }
}

impl Default for FieldType {
    // The default is the most specific type.
    // Type inference proceeds by assuming the most specific type and then
    // relaxing the type as counter-examples are found.
    fn default() -> FieldType { TNull }
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TUnknown => write!(f, "Unknown"),
            TNull => write!(f, "NULL"),
            TUnicode => write!(f, "Unicode"),
            TFloat => write!(f, "Float"),
            TInteger => write!(f, "Integer"),
        }
    }
}

/// TypedSum keeps a rolling sum of the data seen.
///
/// It sums integers until it sees a float, at which point it sums floats.
#[derive(Clone, Default)]
struct TypedSum {
    integer: i64,
    float: Option<f64>,
}

impl TypedSum {
    fn add(&mut self, typ: FieldType, sample: &[u8]) {
        if sample.is_empty() {
            return;
        }
        match typ {
            TFloat => {
                let float: f64 = from_bytes::<f64>(sample).unwrap();
                match self.float {
                    None => {
                        self.float = Some((self.integer as f64) + float);
                    }
                    Some(ref mut f) => {
                        *f += float;
                    }
                }
            }
            TInteger => {
                if let Some(ref mut float) = self.float {
                    *float += from_bytes::<f64>(sample).unwrap();
                } else {
                    self.integer += from_bytes::<i64>(sample).unwrap();
                }
            }
            _ => {}
        }
    }

    fn show(&self, typ: FieldType) -> Option<String> {
        match typ {
            TNull | TUnicode | TUnknown  => None,
            TInteger => Some(self.integer.to_string()),
            TFloat => Some(self.float.unwrap_or(0.0).to_string()),
        }
    }
}

impl Commute for TypedSum {
    fn merge(&mut self, other: TypedSum) {
        match (self.float, other.float) {
            (Some(f1), Some(f2)) => self.float = Some(f1 + f2),
            (Some(f1), None) => self.float = Some(f1 + (other.integer as f64)),
            (None, Some(f2)) => self.float = Some((self.integer as f64) + f2),
            (None, None) => self.integer += other.integer,
        }
    }
}

/// TypedMinMax keeps track of minimum/maximum values for each possible type
/// where min/max makes sense.
#[derive(Clone)]
struct TypedMinMax {
    strings: MinMax<Vec<u8>>,
    str_len: MinMax<usize>,
    integers: MinMax<i64>,
    floats: MinMax<f64>,
}

impl TypedMinMax {
    fn add(&mut self, typ: FieldType, sample: &[u8]) {
        self.str_len.add(sample.len());
        if sample.is_empty() {
            return;
        }
        self.strings.add(sample.to_vec());
        match typ {
            TUnicode | TUnknown | TNull => {}
            TFloat => {
                let n = str::from_utf8(&*sample)
                            .ok()
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap();
                self.floats.add(n);
                self.integers.add(n as i64);
            }
            TInteger => {
                let n = str::from_utf8(&*sample)
                            .ok()
                            .and_then(|s| s.parse::<i64>().ok())
                            .unwrap();
                self.integers.add(n);
                self.floats.add(n as f64);
            }
        }
    }

    fn len_range(&self) -> Option<(String, String)> {
        match (self.str_len.min(), self.str_len.max()) {
            (Some(min), Some(max)) => Some((min.to_string(), max.to_string())),
            _ => None,
        }
    }

    fn show(&self, typ: FieldType) -> Option<(String, String)> {
        match typ {
            TNull => None,
            TUnicode | TUnknown => {
                match (self.strings.min(), self.strings.max()) {
                    (Some(min), Some(max)) => {
                        let min = String::from_utf8_lossy(&**min).to_string();
                        let max = String::from_utf8_lossy(&**max).to_string();
                        Some((min, max))
                    }
                    _ => None
                }
            }
            TInteger => {
                match (self.integers.min(), self.integers.max()) {
                    (Some(min), Some(max)) => {
                        Some((min.to_string(), max.to_string()))
                    }
                    _ => None
                }
            }
            TFloat => {
                match (self.floats.min(), self.floats.max()) {
                    (Some(min), Some(max)) => {
                        Some((min.to_string(), max.to_string()))
                    }
                    _ => None
                }
            }
        }
    }
}

impl Default for TypedMinMax {
    fn default() -> TypedMinMax {
        TypedMinMax {
            strings: Default::default(),
            str_len: Default::default(),
            integers: Default::default(),
            floats: Default::default(),
        }
    }
}

impl Commute for TypedMinMax {
    fn merge(&mut self, other: TypedMinMax) {
        self.strings.merge(other.strings);
        self.str_len.merge(other.str_len);
        self.integers.merge(other.integers);
        self.floats.merge(other.floats);
    }
}

fn from_bytes<T: FromStr>(bytes: &[u8]) -> Option<T> {
    str::from_utf8(bytes).ok().and_then(|s| s.parse().ok())
}

--- MODULE SEPARATOR ---
use std::borrow::Cow;

use csv;
use tabwriter::TabWriter;

use CliResult;
use config::{Config, Delimiter};
use util;

static USAGE: &'static str = "
Outputs CSV data as a table with columns in alignment.

This will not work well if the CSV data contains large fields.

Note that formatting a table requires buffering all CSV data into memory.
Therefore, you should use the 'sample' or 'slice' command to trim down large
CSV data before formatting it with this command.

Usage:
    xsv table [options] [<input>]

table options:
    -w, --width <arg>      The minimum width of each column.
                           [default: 2]
    -p, --pad <arg>        The minimum number of spaces between each column.
                           [default: 2]
    -c, --condense <arg>  Limits the length of each field to the value
                           specified. If the field is UTF-8 encoded, then
                           <arg> refers to the number of code points.
                           Otherwise, it refers to the number of bytes.

Common options:
    -h, --help             Display this message
    -o, --output <file>    Write output to <file> instead of stdout.
    -d, --delimiter <arg>  The field delimiter for reading CSV data.
                           Must be a single character. (default: ,)
";

#[derive(Deserialize)]
struct Args {
    arg_input: Option<String>,
    flag_width: usize,
    flag_pad: usize,
    flag_output: Option<String>,
    flag_delimiter: Option<Delimiter>,
    flag_condense: Option<usize>,
}

pub fn run(argv: &[&str]) -> CliResult<()> {
    let args: Args = util::get_args(USAGE, argv)?;
    let rconfig = Config::new(&args.arg_input)
        .delimiter(args.flag_delimiter)
        .no_headers(true);
    let wconfig = Config::new(&args.flag_output)
        .delimiter(Some(Delimiter(b'\t')));

    let tw = TabWriter::new(wconfig.io_writer()?)
        .minwidth(args.flag_width)
        .padding(args.flag_pad);
    let mut wtr = wconfig.from_writer(tw);
    let mut rdr = rconfig.reader()?;

    let mut record = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut record)? {
        wtr.write_record(record.iter().map(|f| {
            util::condense(Cow::Borrowed(f), args.flag_condense)
        }))?;
    }
    wtr.flush()?;
    Ok(())
}

```

**Filepath**: `./xsv/src/cmd/sort.rs`

---

*Generated by code-ingest export at 2025-09-28 07:48:24 UTC*
