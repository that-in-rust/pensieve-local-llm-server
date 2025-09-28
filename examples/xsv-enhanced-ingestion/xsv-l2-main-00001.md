# Export Row 1

## Metadata

- **Row Index**: 1
- **Total Rows**: 1
- **Exported At**: 2025-09-28T07:48:50.321071+00:00
- **Query Time**: 7ms

## Data

**L2 Window Content**: 

```
# `before_deploy` phase: here we package the build artifacts

set -ex

. $(dirname $0)/utils.sh

# Generate artifacts for release
mk_artifacts() {
    cargo build --target $TARGET --release
}

mk_tarball() {
    # create a "staging" directory
    local td=$(mktempd)
    local out_dir=$(pwd)

    # TODO update this part to copy the artifacts that make sense for your project
    # NOTE All Cargo build artifacts will be under the 'target/$TARGET/{debug,release}'
    cp target/$TARGET/release/xsv $td

    pushd $td

    # release tarball will look like 'rust-everywhere-v1.2.3-x86_64-unknown-linux-gnu.tar.gz'
    tar czf $out_dir/${PROJECT_NAME}-${TRAVIS_TAG}-${TARGET}.tar.gz *

    popd
    rm -r $td
}

main() {
    mk_artifacts
    mk_tarball
}

main

--- MODULE SEPARATOR ---
# `install` phase: install stuff needed for the `script` phase

set -ex

. $(dirname $0)/utils.sh

install_c_toolchain() {
    case $TARGET in
        aarch64-unknown-linux-gnu)
            sudo apt-get install -y --no-install-recommends \
                 gcc-aarch64-linux-gnu libc6-arm64-cross libc6-dev-arm64-cross
            ;;
        *)
            # For other targets, this is handled by addons.apt.packages in .travis.yml
            ;;
    esac
}

install_rustup() {
    curl https://sh.rustup.rs -sSf \
      | sh -s -- -y --default-toolchain=$TRAVIS_RUST_VERSION

    rustc -V
    cargo -V
}

install_standard_crates() {
    if [ $(host) != "$TARGET" ]; then
        rustup target add $TARGET
    fi
}

configure_cargo() {
    local prefix=$(gcc_prefix)

    if [ ! -z $prefix ]; then
        # information about the cross compiler
        ${prefix}gcc -v

        # tell cargo which linker to use for cross compilation
        mkdir -p .cargo
        cat >>.cargo/config <<EOF
[target.$TARGET]
linker = "${prefix}gcc"
EOF
    fi
}

main() {
    install_c_toolchain
    install_rustup
    install_standard_crates
    configure_cargo

    # TODO if you need to install extra stuff add it here
}

main

--- MODULE SEPARATOR ---
# `script` phase: you usually build, test and generate docs in this phase

set -ex

. $(dirname $0)/utils.sh

# NOTE Workaround for rust-lang/rust#31907 - disable doc tests when cross compiling
# This has been fixed in the nightly channel but it would take a while to reach the other channels
disable_cross_doctests() {
    if [ $(host) != "$TARGET" ] && [ "$TRAVIS_RUST_VERSION" = "stable" ]; then
        if [ "$TRAVIS_OS_NAME" = "osx" ]; then
            brew install gnu-sed --default-names
        fi

        find src -name '*.rs' -type f | xargs sed -i -e 's:\(//.\s*```\):\1 ignore,:g'
    fi
}

# TODO modify this function as you see fit
# PROTIP Always pass `--target $TARGET` to cargo commands, this makes cargo output build artifacts
# to target/$TARGET/{debug,release} which can reduce the number of needed conditionals in the
# `before_deploy`/packaging phase
run_test_suite() {
    case $TARGET in
        # configure emulation for transparent execution of foreign binaries
        aarch64-unknown-linux-gnu)
            export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
            ;;
        arm*-unknown-linux-gnueabihf)
            export QEMU_LD_PREFIX=/usr/arm-linux-gnueabihf
            ;;
        *)
            ;;
    esac

    if [ ! -z "$QEMU_LD_PREFIX" ]; then
        # Run tests on a single thread when using QEMU user emulation
        export RUST_TEST_THREADS=1
    fi

    cargo build --target $TARGET --verbose
    cargo test --target $TARGET

    # sanity check the file type
    file target/$TARGET/debug/xsv
}

main() {
    disable_cross_doctests
    run_test_suite
}

main

--- MODULE SEPARATOR ---
mktempd() {
    echo $(mktemp -d 2>/dev/null || mktemp -d -t tmp)
}

host() {
    case "$TRAVIS_OS_NAME" in
        linux)
            echo x86_64-unknown-linux-gnu
            ;;
        osx)
            echo x86_64-apple-darwin
            ;;
    esac
}

gcc_prefix() {
    case "$TARGET" in
        aarch64-unknown-linux-gnu)
            echo aarch64-linux-gnu-
            ;;
        arm*-gnueabihf)
            echo arm-linux-gnueabihf-
            ;;
        *)
            return
            ;;
    esac
}

dobin() {
    [ -z $MAKE_DEB ] && die 'dobin: $MAKE_DEB not set'
    [ $# -lt 1 ] && die "dobin: at least one argument needed"

    local f prefix=$(gcc_prefix)
    for f in "$@"; do
        install -m0755 $f $dtd/debian/usr/bin/
        ${prefix}strip -s $dtd/debian/usr/bin/$(basename $f)
    done
}

architecture() {
    case $1 in
        x86_64-unknown-linux-gnu|x86_64-unknown-linux-musl)
            echo amd64
            ;;
        i686-unknown-linux-gnu|i686-unknown-linux-musl)
            echo i386
            ;;
        arm*-unknown-linux-gnueabihf)
            echo armhf
            ;;
        *)
            die "architecture: unexpected target $TARGET"
            ;;
    esac
}

--- MODULE SEPARATOR ---
#[allow(deprecated, unused_imports)]
use std::ascii::AsciiExt;
use std::borrow::ToOwned;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::ops::Deref;
use std::path::PathBuf;

use csv;
use index::Indexed;
use serde::de::{Deserializer, Deserialize, Error};

use CliResult;
use select::{SelectColumns, Selection};
use util;


#[derive(Clone, Copy, Debug)]
pub struct Delimiter(pub u8);

/// Delimiter represents values that can be passed from the command line that
/// can be used as a field delimiter in CSV data.
///
/// Its purpose is to ensure that the Unicode character given decodes to a
/// valid ASCII character as required by the CSV parser.
impl Delimiter {
    pub fn as_byte(self) -> u8 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Delimiter {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Delimiter, D::Error> {
        let c = String::deserialize(d)?;
        match &*c {
            r"\t" => Ok(Delimiter(b'\t')),
            s => {
                if s.len() != 1 {
                    let msg = format!("Could not convert '{}' to a single \
                                       ASCII character.", s);
                    return Err(D::Error::custom(msg));
                }
                let c = s.chars().next().unwrap();
                if c.is_ascii() {
                    Ok(Delimiter(c as u8))
                } else {
                    let msg = format!("Could not convert '{}' \
                                       to ASCII delimiter.", c);
                    Err(D::Error::custom(msg))
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Config {
    path: Option<PathBuf>, // None implies <stdin>
    idx_path: Option<PathBuf>,
    select_columns: Option<SelectColumns>,
    delimiter: u8,
    pub no_headers: bool,
    flexible: bool,
    terminator: csv::Terminator,
    quote: u8,
    quote_style: csv::QuoteStyle,
    double_quote: bool,
    escape: Option<u8>,
    quoting: bool,
}

impl Config {
    pub fn new(path: &Option<String>) -> Config {
        let (path, delim) = match *path {
            None => (None, b','),
            Some(ref s) if s.deref() == "-" => (None, b','),
            Some(ref s) => {
                let path = PathBuf::from(s);
                let delim =
                    if path.extension().map_or(false, |v| v == "tsv" || v == "tab") {
                        b'\t'
                    } else {
                        b','
                    };
                (Some(path), delim)
            }
        };
        Config {
            path: path,
            idx_path: None,
            select_columns: None,
            delimiter: delim,
            no_headers: false,
            flexible: false,
            terminator: csv::Terminator::Any(b'\n'),
            quote: b'"',
            quote_style: csv::QuoteStyle::Necessary,
            double_quote: true,
            escape: None,
            quoting: true,
        }
    }

    pub fn delimiter(mut self, d: Option<Delimiter>) -> Config {
        if let Some(d) = d {
            self.delimiter = d.as_byte();
        }
        self
    }

    pub fn no_headers(mut self, mut yes: bool) -> Config {
        if env::var("XSV_TOGGLE_HEADERS").unwrap_or("0".to_owned()) == "1" {
            yes = !yes;
        }
        self.no_headers = yes;
        self
    }

    pub fn flexible(mut self, yes: bool) -> Config {
        self.flexible = yes;
        self
    }

    pub fn crlf(mut self, yes: bool) -> Config {
        if yes {
            self.terminator = csv::Terminator::CRLF;
        } else {
            self.terminator = csv::Terminator::Any(b'\n');
        }
        self
    }

    pub fn terminator(mut self, term: csv::Terminator) -> Config {
        self.terminator = term;
        self
    }

    pub fn quote(mut self, quote: u8) -> Config {
        self.quote = quote;
        self
    }

    pub fn quote_style(mut self, style: csv::QuoteStyle) -> Config {
        self.quote_style = style;
        self
    }

    pub fn double_quote(mut self, yes: bool) -> Config {
        self.double_quote = yes;
        self
    }

    pub fn escape(mut self, escape: Option<u8>) -> Config {
        self.escape = escape;
        self
    }

    pub fn quoting(mut self, yes: bool) -> Config {
        self.quoting = yes;
        self
    }

    pub fn select(mut self, sel_cols: SelectColumns) -> Config {
        self.select_columns = Some(sel_cols);
        self
    }

    pub fn is_std(&self) -> bool {
        self.path.is_none()
    }

    pub fn selection(
        &self,
        first_record: &csv::ByteRecord,
    ) -> Result<Selection, String> {
        match self.select_columns {
            None => Err("Config has no 'SelectColums'. Did you call \
                         Config::select?".to_owned()),
            Some(ref sel) => sel.selection(first_record, !self.no_headers),
        }
    }

    pub fn write_headers<R: io::Read, W: io::Write>
                        (&self, r: &mut csv::Reader<R>, w: &mut csv::Writer<W>)
                        -> csv::Result<()> {
        if !self.no_headers {
            let r = r.byte_headers()?;
            if !r.is_empty() {
                w.write_record(r)?;
            }
        }
        Ok(())
    }

    pub fn writer(&self)
                 -> io::Result<csv::Writer<Box<io::Write+'static>>> {
        Ok(self.from_writer(self.io_writer()?))
    }

    pub fn reader(&self)
                 -> io::Result<csv::Reader<Box<io::Read+'static>>> {
        Ok(self.from_reader(self.io_reader()?))
    }

    pub fn reader_file(&self) -> io::Result<csv::Reader<fs::File>> {
        match self.path {
            None => Err(io::Error::new(
                io::ErrorKind::Other, "Cannot use <stdin> here",
            )),
            Some(ref p) => fs::File::open(p).map(|f| self.from_reader(f)),
        }
    }

    pub fn index_files(&self)
           -> io::Result<Option<(csv::Reader<fs::File>, fs::File)>> {
        let (csv_file, idx_file) = match (&self.path, &self.idx_path) {
            (&None, &None) => return Ok(None),
            (&None, &Some(_)) => return Err(io::Error::new(
                io::ErrorKind::Other,
                "Cannot use <stdin> with indexes",
                // Some(format!("index file: {}", p.display()))
            )),
            (&Some(ref p), &None) => {
                // We generally don't want to report an error here, since we're
                // passively trying to find an index.
                let idx_file = match fs::File::open(&util::idx_path(p)) {
                    // TODO: Maybe we should report an error if the file exists
                    // but is not readable.
                    Err(_) => return Ok(None),
                    Ok(f) => f,
                };
                (fs::File::open(p)?, idx_file)
            }
            (&Some(ref p), &Some(ref ip)) => {
                (fs::File::open(p)?, fs::File::open(ip)?)
            }
        };
        // If the CSV data was last modified after the index file was last
        // modified, then return an error and demand the user regenerate the
        // index.
        let data_modified = util::last_modified(&csv_file.metadata()?);
        let idx_modified = util::last_modified(&idx_file.metadata()?);
        if data_modified > idx_modified {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "The CSV file was modified after the index file. \
                 Please re-create the index.",
            ));
        }
        let csv_rdr = self.from_reader(csv_file);
        Ok(Some((csv_rdr, idx_file)))
    }

    pub fn indexed(&self)
                  -> CliResult<Option<Indexed<fs::File, fs::File>>> {
        match self.index_files()? {
            None => Ok(None),
            Some((r, i)) => Ok(Some(Indexed::open(r, i)?)),
        }
    }

    pub fn io_reader(&self) -> io::Result<Box<io::Read+'static>> {
        Ok(match self.path {
                None => Box::new(io::stdin()),
                Some(ref p) => {
                    match fs::File::open(p){
                        Ok(x) => Box::new(x),
                        Err(err) => {
                            let msg = format!(
                                "failed to open {}: {}", p.display(), err);
                            return Err(io::Error::new(
                                io::ErrorKind::NotFound,
                                msg,
                            ));
                        }
                    }
                },
            })
    }

    pub fn from_reader<R: Read>(&self, rdr: R) -> csv::Reader<R> {
        csv::ReaderBuilder::new()
            .flexible(self.flexible)
            .delimiter(self.delimiter)
            .has_headers(!self.no_headers)
            .quote(self.quote)
            .quoting(self.quoting)
            .escape(self.escape)
            .from_reader(rdr)
    }

    pub fn io_writer(&self) -> io::Result<Box<io::Write+'static>> {
        Ok(match self.path {
            None => Box::new(io::stdout()),
            Some(ref p) => Box::new(fs::File::create(p)?),
        })
    }

    pub fn from_writer<W: io::Write>(&self, wtr: W) -> csv::Writer<W> {
        csv::WriterBuilder::new()
            .flexible(self.flexible)
            .delimiter(self.delimiter)
            .terminator(self.terminator)
            .quote(self.quote)
            .quote_style(self.quote_style)
            .double_quote(self.double_quote)
            .escape(self.escape.unwrap_or(b'\\'))
            .buffer_capacity(32 * (1<<10))
            .from_writer(wtr)
    }
}

--- MODULE SEPARATOR ---
use std::io;
use std::ops;

use csv;
use csv_index::RandomAccessSimple;

use CliResult;

/// Indexed composes a CSV reader with a simple random access index.
pub struct Indexed<R, I> {
    csv_rdr: csv::Reader<R>,
    idx: RandomAccessSimple<I>,
}

impl<R, I> ops::Deref for Indexed<R, I> {
    type Target = csv::Reader<R>;
    fn deref(&self) -> &csv::Reader<R> { &self.csv_rdr }
}

impl<R, I> ops::DerefMut for Indexed<R, I> {
    fn deref_mut(&mut self) -> &mut csv::Reader<R> { &mut self.csv_rdr }
}

impl<R: io::Read + io::Seek, I: io::Read + io::Seek> Indexed<R, I> {
    /// Opens an index.
    pub fn open(
        csv_rdr: csv::Reader<R>,
        idx_rdr: I,
    ) -> CliResult<Indexed<R, I>> {
        Ok(Indexed {
            csv_rdr: csv_rdr,
            idx: RandomAccessSimple::open(idx_rdr)?,
        })
    }

    /// Return the number of records (not including the header record) in this
    /// index.
    pub fn count(&self) -> u64 {
        if self.csv_rdr.has_headers() && !self.idx.is_empty() {
            self.idx.len() - 1
        } else {
            self.idx.len()
        }
    }

    /// Seek to the starting position of record `i`.
    pub fn seek(&mut self, mut i: u64) -> CliResult<()> {
        if i >= self.count() {
            let msg = format!(
                "invalid record index {} (there are {} records)",
                i, self.count());
            return fail!(io::Error::new(io::ErrorKind::Other, msg));
        }
        if self.csv_rdr.has_headers() {
            i += 1;
        }
        let pos = self.idx.get(i)?;
        self.csv_rdr.seek(pos)?;
        Ok(())
    }
}

--- MODULE SEPARATOR ---
extern crate byteorder;
extern crate crossbeam_channel as channel;
extern crate csv;
extern crate csv_index;
extern crate docopt;
extern crate filetime;
extern crate num_cpus;
extern crate rand;
extern crate regex;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate stats;
extern crate tabwriter;
extern crate threadpool;

use std::borrow::ToOwned;
use std::env;
use std::fmt;
use std::io;
use std::process;

use docopt::Docopt;

macro_rules! wout {
    ($($arg:tt)*) => ({
        use std::io::Write;
        (writeln!(&mut ::std::io::stdout(), $($arg)*)).unwrap();
    });
}

macro_rules! werr {
    ($($arg:tt)*) => ({
        use std::io::Write;
        (writeln!(&mut ::std::io::stderr(), $($arg)*)).unwrap();
    });
}

macro_rules! fail {
    ($e:expr) => (Err(::std::convert::From::from($e)));
}

macro_rules! command_list {
    () => (
"
    cat         Concatenate by row or column
    count       Count records
    fixlengths  Makes all records have same length
    flatten     Show one field per line
    fmt         Format CSV output (change field delimiter)
    frequency   Show frequency tables
    headers     Show header names
    help        Show this usage message.
    index       Create CSV index for faster access
    input       Read CSV data with special quoting rules
    join        Join CSV files
    partition   Partition CSV data based on a column value
    sample      Randomly sample CSV data
    reverse     Reverse rows of CSV data
    search      Search CSV data with regexes
    select      Select columns from CSV
    slice       Slice records from CSV
    sort        Sort CSV data
    split       Split CSV data into many files
    stats       Compute basic statistics
    table       Align CSV data into columns
"
    )
}

mod cmd;
mod config;
mod index;
mod select;
mod util;

static USAGE: &'static str = concat!("
Usage:
    xsv <command> [<args>...]
    xsv [options]

Options:
    --list        List all commands available.
    -h, --help    Display this message
    <command> -h  Display the command help message
    --version     Print version info and exit

Commands:", command_list!());

#[derive(Deserialize)]
struct Args {
    arg_command: Option<Command>,
    flag_list: bool,
}

fn main() {
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.options_first(true)
                                           .version(Some(util::version()))
                                           .deserialize())
                            .unwrap_or_else(|e| e.exit());
    if args.flag_list {
        wout!(concat!("Installed commands:", command_list!()));
        return;
    }
    match args.arg_command {
        None => {
            werr!(concat!(
                "xsv is a suite of CSV command line utilities.

Please choose one of the following commands:",
                command_list!()));
            process::exit(0);
        }
        Some(cmd) => {
            match cmd.run() {
                Ok(()) => process::exit(0),
                Err(CliError::Flag(err)) => err.exit(),
                Err(CliError::Csv(err)) => {
                    werr!("{}", err);
                    process::exit(1);
                }
                Err(CliError::Io(ref err))
                        if err.kind() == io::ErrorKind::BrokenPipe => {
                    process::exit(0);
                }
                Err(CliError::Io(err)) => {
                    werr!("{}", err);
                    process::exit(1);
                }
                Err(CliError::Other(msg)) => {
                    werr!("{}", msg);
                    process::exit(1);
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Command {
    Cat,
    Count,
    FixLengths,
    Flatten,
    Fmt,
    Frequency,
    Headers,
    Help,
    Index,
    Input,
    Join,
    Partition,
    Reverse,
    Sample,
    Search,
    Select,
    Slice,
    Sort,
    Split,
    Stats,
    Table,
}

impl Command {
    fn run(self) -> CliResult<()> {
        let argv: Vec<_> = env::args().map(|v| v.to_owned()).collect();
        let argv: Vec<_> = argv.iter().map(|s| &**s).collect();
        let argv = &*argv;

        if !argv[1].chars().all(char::is_lowercase) {
            return Err(CliError::Other(format!(
                "xsv expects commands in lowercase. Did you mean '{}'?", 
                argv[1].to_lowercase()).to_string()));
        }
        match self {
            Command::Cat => cmd::cat::run(argv),
            Command::Count => cmd::count::run(argv),
            Command::FixLengths => cmd::fixlengths::run(argv),
            Command::Flatten => cmd::flatten::run(argv),
            Command::Fmt => cmd::fmt::run(argv),
            Command::Frequency => cmd::frequency::run(argv),
            Command::Headers => cmd::headers::run(argv),
            Command::Help => { wout!("{}", USAGE); Ok(()) }
            Command::Index => cmd::index::run(argv),
            Command::Input => cmd::input::run(argv),
            Command::Join => cmd::join::run(argv),
            Command::Partition => cmd::partition::run(argv),
            Command::Reverse => cmd::reverse::run(argv),
            Command::Sample => cmd::sample::run(argv),
            Command::Search => cmd::search::run(argv),
            Command::Select => cmd::select::run(argv),
            Command::Slice => cmd::slice::run(argv),
            Command::Sort => cmd::sort::run(argv),
            Command::Split => cmd::split::run(argv),
            Command::Stats => cmd::stats::run(argv),
            Command::Table => cmd::table::run(argv),
        }
    }
}

pub type CliResult<T> = Result<T, CliError>;

#[derive(Debug)]
pub enum CliError {
    Flag(docopt::Error),
    Csv(csv::Error),
    Io(io::Error),
    Other(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CliError::Flag(ref e) => { e.fmt(f) }
            CliError::Csv(ref e) => { e.fmt(f) }
            CliError::Io(ref e) => { e.fmt(f) }
            CliError::Other(ref s) => { f.write_str(&**s) }
        }
    }
}

impl From<docopt::Error> for CliError {
    fn from(err: docopt::Error) -> CliError {
        CliError::Flag(err)
    }
}

impl From<csv::Error> for CliError {
    fn from(err: csv::Error) -> CliError {
        if !err.is_io_error() {
            return CliError::Csv(err);
        }
        match err.into_kind() {
            csv::ErrorKind::Io(v) => From::from(v),
            _ => unreachable!(),
        }
    }
}

impl From<io::Error> for CliError {
    fn from(err: io::Error) -> CliError {
        CliError::Io(err)
    }
}

impl From<String> for CliError {
    fn from(err: String) -> CliError {
        CliError::Other(err)
    }
}

impl<'a> From<&'a str> for CliError {
    fn from(err: &'a str) -> CliError {
        CliError::Other(err.to_owned())
    }
}

impl From<regex::Error> for CliError {
    fn from(err: regex::Error) -> CliError {
        CliError::Other(format!("{:?}", err))
    }
}

--- MODULE SEPARATOR ---
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::iter::{self, repeat};
use std::ops;
use std::slice;
use std::str::FromStr;

use csv;
use serde::de::{Deserializer, Deserialize, Error};

#[derive(Clone)]
pub struct SelectColumns {
    selectors: Vec<Selector>,
    invert: bool,
}

impl SelectColumns {
    fn parse(mut s: &str) -> Result<SelectColumns, String> {
        let invert =
            if !s.is_empty() && s.as_bytes()[0] == b'!' {
                s = &s[1..];
                true
            } else {
                false
            };
        Ok(SelectColumns {
            selectors: SelectorParser::new(s).parse()?,
            invert: invert,
        })
    }

    pub fn selection(
        &self,
        first_record: &csv::ByteRecord,
        use_names: bool,
    ) -> Result<Selection, String> {
        if self.selectors.is_empty() {
            return Ok(Selection(if self.invert {
                // Inverting everything means we get nothing.
                vec![]
            } else {
                (0..first_record.len()).collect()
            }));
        }

        let mut map = vec![];
        for sel in &self.selectors {
            let idxs = sel.indices(first_record, use_names);
            map.extend(idxs?.into_iter());
        }
        if self.invert {
            let set: HashSet<_> = map.into_iter().collect();
            let mut map = vec![];
            for i in 0..first_record.len() {
                if !set.contains(&i) {
                    map.push(i);
                }
            }
            return Ok(Selection(map));
        }
        Ok(Selection(map))
    }
}

impl fmt::Debug for SelectColumns {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.selectors.is_empty() {
            write!(f, "<All>")
        } else {
            let strs: Vec<_> =
                self.selectors
                    .iter().map(|sel| format!("{:?}", sel)).collect();
            write!(f, "{}", strs.join(", "))
        }
    }
}

impl<'de> Deserialize<'de> for SelectColumns {
    fn deserialize<D: Deserializer<'de>>(
        d: D,
    ) -> Result<SelectColumns, D::Error> {
        let raw = String::deserialize(d)?;
        SelectColumns::parse(&raw).map_err(|e| D::Error::custom(&e))
    }
}

struct SelectorParser {
    chars: Vec<char>,
    pos: usize,
}

impl SelectorParser {
    fn new(s: &str) -> SelectorParser {
        SelectorParser { chars: s.chars().collect(), pos: 0 }
    }

    fn parse(&mut self) -> Result<Vec<Selector>, String> {
        let mut sels = vec![];
        loop {
            if self.cur().is_none() {
                break;
            }
            let f1: OneSelector =
                if self.cur() == Some('-') {
                    OneSelector::Start
                } else {
                    self.parse_one()?
                };
            let f2: Option<OneSelector> =
                if self.cur() == Some('-') {
                    self.bump();
                    Some(if self.is_end_of_selector() {
                        OneSelector::End
                    } else {
                        self.parse_one()?
                    })
                } else {
                    None
                };
            if !self.is_end_of_selector() {
                return Err(format!(
                    "Expected end of field but got '{}' instead.",
                    self.cur().unwrap()));
            }
            sels.push(match f2 {
                Some(end) => Selector::Range(f1, end),
                None => Selector::One(f1),
            });
            self.bump();
        }
        Ok(sels)
    }

    fn parse_one(&mut self) -> Result<OneSelector, String> {
        let name =
            if self.cur() == Some('"') {
                self.bump();
                self.parse_quoted_name()?
            } else {
                self.parse_name()?
            };
        Ok(if self.cur() == Some('[') {
            let idx = self.parse_index()?;
            OneSelector::IndexedName(name, idx)
        } else {
            match FromStr::from_str(&name) {
                Err(_) => OneSelector::IndexedName(name, 0),
                Ok(idx) => OneSelector::Index(idx),
            }
        })
    }

    fn parse_name(&mut self) -> Result<String, String> {
        let mut name = String::new();
        loop {
            if self.is_end_of_field() || self.cur() == Some('[') {
                break;
            }
            name.push(self.cur().unwrap());
            self.bump();
        }
        Ok(name)
    }

    fn parse_quoted_name(&mut self) -> Result<String, String> {
        let mut name = String::new();
        loop {
            match self.cur() {
                None => {
                    return Err("Unclosed quote, missing closing \"."
                               .to_owned());
                }
                Some('"') => {
                    self.bump();
                    if self.cur() == Some('"') {
                        self.bump();
                        name.push('"'); name.push('"');
                        continue;
                    }
                    break
                }
                Some(c) => { name.push(c); self.bump(); }
            }
        }
        Ok(name)
    }

    fn parse_index(&mut self) -> Result<usize, String> {
        assert_eq!(self.cur().unwrap(), '[');
        self.bump();

        let mut idx = String::new();
        loop {
            match self.cur() {
                None => {
                    return Err("Unclosed index bracket, missing closing ]."
                               .to_owned());
                }
                Some(']') => { self.bump(); break; }
                Some(c) => { idx.push(c); self.bump(); }
            }
        }
        FromStr::from_str(&idx).map_err(|err| {
            format!("Could not convert '{}' to an integer: {}", idx, err)
        })
    }

    fn cur(&self) -> Option<char> {
        self.chars.get(self.pos).cloned()
    }

    fn is_end_of_field(&self) -> bool {
        self.cur().map_or(true, |c| c == ',' || c == '-')
    }

    fn is_end_of_selector(&self) -> bool {
        self.cur().map_or(true, |c| c == ',')
    }

    fn bump(&mut self) {
        if self.pos < self.chars.len() { self.pos += 1; }
    }
}

#[derive(Clone)]
enum Selector {
    One(OneSelector),
    Range(OneSelector, OneSelector),
}

#[derive(Clone)]
enum OneSelector {
    Start,
    End,
    Index(usize),
    IndexedName(String, usize),
}

impl Selector {
    fn indices(
        &self,
        first_record: &csv::ByteRecord,
        use_names: bool,
    ) -> Result<Vec<usize>, String> {
        match *self {
            Selector::One(ref sel) => {
                sel.index(first_record, use_names).map(|i| vec![i])
            }
            Selector::Range(ref sel1, ref sel2) => {
                let i1 = sel1.index(first_record, use_names)?;
                let i2 = sel2.index(first_record, use_names)?;
                Ok(match i1.cmp(&i2) {
                    Ordering::Equal => vec!(i1),
                    Ordering::Less => (i1..(i2 + 1)).collect(),
                    Ordering::Greater => {
                        let mut inds = vec![];
                        let mut i = i1 + 1;
                        while i > i2 {
                            i -= 1;
                            inds.push(i);
                        }
                        inds
                    }
                })
            }
        }
    }
}

impl OneSelector {
    fn index(
        &self,
        first_record: &csv::ByteRecord,
        use_names: bool,
    ) -> Result<usize, String> {
        match *self {
            OneSelector::Start => Ok(0),
            OneSelector::End => Ok(
                if first_record.len() == 0 {
                    0
                } else {
                    first_record.len() - 1
                }
            ),
            OneSelector::Index(i) => {
                if i < 1 || i > first_record.len() {
                    Err(format!("Selector index {} is out of \
                                 bounds. Index must be >= 1 \
                                 and <= {}.", i, first_record.len()))
                } else {
                    // Indices given by user are 1-offset. Convert them here!
                    Ok(i-1)
                }
            }
            OneSelector::IndexedName(ref s, sidx) => {
                if !use_names {
                    return Err(format!("Cannot use names ('{}') in selection \
                                        with --no-headers set.", s));
                }
                let mut num_found = 0;
                for (i, field) in first_record.iter().enumerate() {
                    if field == s.as_bytes() {
                        if num_found == sidx {
                            return Ok(i);
                        }
                        num_found += 1;
                    }
                }
                if num_found == 0 {
                    Err(format!("Selector name '{}' does not exist \
                                 as a named header in the given CSV \
                                 data.", s))
                } else {
                    Err(format!("Selector index '{}' for name '{}' is \
                                 out of bounds. Must be >= 0 and <= {}.",
                                 sidx, s, num_found - 1))
                }
            }
        }
    }
}

impl fmt::Debug for Selector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Selector::One(ref sel) => sel.fmt(f),
            Selector::Range(ref s, ref e) =>
                write!(f, "Range({:?}, {:?})", s, e),
        }
    }
}

impl fmt::Debug for OneSelector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OneSelector::Start => write!(f, "Start"),
            OneSelector::End => write!(f, "End"),
            OneSelector::Index(idx) => write!(f, "Index({})", idx),
            OneSelector::IndexedName(ref s, idx) =>
                write!(f, "IndexedName({}[{}])", s, idx),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Selection(Vec<usize>);

pub type _GetField =
    for <'c> fn(&mut &'c csv::ByteRecord, &usize) -> Option<&'c [u8]>;

impl Selection {
    pub fn select<'a, 'b>(&'a self, row: &'b csv::ByteRecord)
                 -> iter::Scan<
                        slice::Iter<'a, usize>,
                        &'b csv::ByteRecord,
                        _GetField,
                    > {
        // This is horrifying.
        fn get_field<'c>(row: &mut &'c csv::ByteRecord, idx: &usize)
                        -> Option<&'c [u8]> {
            Some(&row[*idx])
        }
        let get_field: _GetField = get_field;
        self.iter().scan(row, get_field)
    }

    pub fn normal(&self) -> NormalSelection {
        let &Selection(ref inds) = self;
        if inds.is_empty() {
            return NormalSelection(vec![]);
        }

        let mut normal = inds.clone();
        normal.sort();
        normal.dedup();
        let mut set: Vec<_> =
            repeat(false).take(normal[normal.len()-1] + 1).collect();
        for i in normal.into_iter() {
            set[i] = true;
        }
        NormalSelection(set)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl ops::Deref for Selection {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct NormalSelection(Vec<bool>);

pub type _NormalScan<'a, T, I> = iter::Scan<
    iter::Enumerate<I>,
    &'a [bool],
    _NormalGetField<T>,
>;

pub type _NormalFilterMap<'a, T, I> = iter::FilterMap<
    _NormalScan<'a, T, I>,
    fn(Option<T>) -> Option<T>
>;

pub type _NormalGetField<T> =
    fn(&mut &[bool], (usize, T)) -> Option<Option<T>>;

impl NormalSelection {
    pub fn select<'a, T, I>(&'a self, row: I) -> _NormalFilterMap<'a, T, I>
             where I: Iterator<Item=T> {
        fn filmap<T>(v: Option<T>) -> Option<T> { v }
        fn get_field<T>(set: &mut &[bool], t: (usize, T))
                       -> Option<Option<T>> {
            let (i, v) = t;
            if i < set.len() && set[i] { Some(Some(v)) } else { Some(None) }
        }
        let get_field: _NormalGetField<T> = get_field;
        let filmap: fn(Option<T>) -> Option<T> = filmap;
        row.enumerate().scan(&**self, get_field).filter_map(filmap)
    }

    pub fn len(&self) -> usize {
        self.iter().filter(|b| **b).count()
    }
}

impl ops::Deref for NormalSelection {
    type Target = [bool];

    fn deref(&self) -> &[bool] {
        &self.0
    }
}

--- MODULE SEPARATOR ---
use std::borrow::Cow;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str;
use std::thread;
use std::time;

use csv;
use docopt::Docopt;
use num_cpus;
use serde::de::{Deserializer, Deserialize, DeserializeOwned, Error};

use CliResult;
use config::{Config, Delimiter};

pub fn num_cpus() -> usize {
    num_cpus::get()
}

pub fn version() -> String {
    let (maj, min, pat) = (
        option_env!("CARGO_PKG_VERSION_MAJOR"),
        option_env!("CARGO_PKG_VERSION_MINOR"),
        option_env!("CARGO_PKG_VERSION_PATCH"),
    );
    match (maj, min, pat) {
        (Some(maj), Some(min), Some(pat)) =>
            format!("{}.{}.{}", maj, min, pat),
        _ => "".to_owned(),
    }
}

pub fn get_args<T>(usage: &str, argv: &[&str]) -> CliResult<T>
        where T: DeserializeOwned {
    Docopt::new(usage)
           .and_then(|d| d.argv(argv.iter().map(|&x| x))
                          .version(Some(version()))
                          .deserialize())
           .map_err(From::from)
}

pub fn many_configs(inps: &[String], delim: Option<Delimiter>,
                    no_headers: bool) -> Result<Vec<Config>, String> {
    let mut inps = inps.to_vec();
    if inps.is_empty() {
        inps.push("-".to_owned()); // stdin
    }
    let confs = inps.into_iter()
                    .map(|p| Config::new(&Some(p))
                                    .delimiter(delim)
                                    .no_headers(no_headers))
                    .collect::<Vec<_>>();
    errif_greater_one_stdin(&*confs)?;
    Ok(confs)
}

pub fn errif_greater_one_stdin(inps: &[Config]) -> Result<(), String> {
    let nstd = inps.iter().filter(|inp| inp.is_std()).count();
    if nstd > 1 {
        return Err("At most one <stdin> input is allowed.".to_owned());
    }
    Ok(())
}

pub fn chunk_size(nitems: usize, njobs: usize) -> usize {
    if nitems < njobs {
        nitems
    } else {
        nitems / njobs
    }
}

pub fn num_of_chunks(nitems: usize, chunk_size: usize) -> usize {
    if chunk_size == 0 {
        return nitems;
    }
    let mut n = nitems / chunk_size;
    if nitems % chunk_size != 0 {
        n += 1;
    }
    n
}

pub fn last_modified(md: &fs::Metadata) -> u64 {
    use filetime::FileTime;
    FileTime::from_last_modification_time(md).seconds_relative_to_1970()
}

pub fn condense<'a>(val: Cow<'a, [u8]>, n: Option<usize>) -> Cow<'a, [u8]> {
    match n {
        None => val,
        Some(n) => {
            let mut is_short_utf8 = false;
            if let Ok(s) = str::from_utf8(&*val) {
                if n >= s.chars().count() {
                    is_short_utf8 = true;
                } else {
                    let mut s = s.chars().take(n).collect::<String>();
                    s.push_str("...");
                    return Cow::Owned(s.into_bytes());
                }
            }
            if is_short_utf8 || n >= (*val).len() { // already short enough
                val
            } else {
                // This is a non-Unicode string, so we just trim on bytes.
                let mut s = val[0..n].to_vec();
                s.extend(b"...".iter().cloned());
                Cow::Owned(s)
            }
        }
    }
}

pub fn idx_path(csv_path: &Path) -> PathBuf {
    let mut p = csv_path.to_path_buf().into_os_string().into_string().unwrap();
    p.push_str(".idx");
    PathBuf::from(&p)
}

pub type Idx = Option<usize>;

pub fn range(start: Idx, end: Idx, len: Idx, index: Idx)
            -> Result<(usize, usize), String> {
    match (start, end, len, index) {
        (None, None, None, Some(i)) => Ok((i, i+1)),
        (_, _, _, Some(_)) =>
            Err("--index cannot be used with --start, --end or --len".to_owned()),
        (_, Some(_), Some(_), None) =>
            Err("--end and --len cannot be used at the same time.".to_owned()),
        (_, None, None, None) => Ok((start.unwrap_or(0), ::std::usize::MAX)),
        (_, Some(e), None, None) => {
            let s = start.unwrap_or(0);
            if s > e {
                Err(format!("The end of the range ({}) must be greater than or\n\
                             equal to the start of the range ({}).", e, s))
            } else {
                Ok((s, e))
            }
        }
        (_, None, Some(l), None) => {
            let s = start.unwrap_or(0);
            Ok((s, s + l))
        }
    }
}

/// Create a directory recursively, avoiding the race conditons fixed by
/// https://github.com/rust-lang/rust/pull/39799.
fn create_dir_all_threadsafe(path: &Path) -> io::Result<()> {
    // Try 20 times. This shouldn't theoretically need to be any larger
    // than the number of nested directories we need to create.
    for _ in 0..20 {
        match fs::create_dir_all(path) {
            // This happens if a directory in `path` doesn't exist when we
            // test for it, and another thread creates it before we can.
            Err(ref err) if err.kind() == io::ErrorKind::AlreadyExists => {},
            other => return other,
        }
        // We probably don't need to sleep at all, because the intermediate
        // directory is already created.  But let's attempt to back off a
        // bit and let the other thread finish.
        thread::sleep(time::Duration::from_millis(25));
    }
    // Try one last time, returning whatever happens.
    fs::create_dir_all(path)
}

/// Represents a filename template of the form `"{}.csv"`, where `"{}"` is
/// the splace to insert the part of the filename generated by `xsv`.
#[derive(Clone, Debug)]
pub struct FilenameTemplate {
    prefix: String,
    suffix: String,
}

impl FilenameTemplate {
    /// Generate a new filename using `unique_value` to replace the `"{}"`
    /// in the template.
    pub fn filename(&self, unique_value: &str) -> String {
        format!("{}{}{}", &self.prefix, unique_value, &self.suffix)
    }

    /// Create a new, writable file in directory `path` with a filename
    /// using `unique_value` to replace the `"{}"` in the template.  Note
    /// that we do not output headers; the caller must do that if
    /// desired.
    pub fn writer<P>(&self, path: P, unique_value: &str)
                 -> io::Result<csv::Writer<Box<io::Write+'static>>>
        where P: AsRef<Path>
    {
        let filename = self.filename(unique_value);
        let full_path = path.as_ref().join(filename);
        if let Some(parent) = full_path.parent() {
            // We may be called concurrently, especially by parallel `xsv
            // split`, so be careful to avoid the `create_dir_all` race
            // condition.
            create_dir_all_threadsafe(parent)?;
        }
        let spath = Some(full_path.display().to_string());
        Config::new(&spath).writer()
    }
}

impl<'de> Deserialize<'de> for FilenameTemplate {
    fn deserialize<D: Deserializer<'de>>(
        d: D,
    ) -> Result<FilenameTemplate, D::Error> {
        let raw = String::deserialize(d)?;
        let chunks = raw.split("{}").collect::<Vec<_>>();
        if chunks.len() == 2 {
            Ok(FilenameTemplate {
                prefix: chunks[0].to_owned(),
                suffix: chunks[1].to_owned(),
            })
        } else {
            Err(D::Error::custom(
                "The --filename argument must contain one '{}'."))
        }
    }
}

--- MODULE SEPARATOR ---
use std::process;

use {Csv, CsvData, qcheck};
use workdir::Workdir;

fn no_headers(cmd: &mut process::Command) {
    cmd.arg("--no-headers");
}

fn pad(cmd: &mut process::Command) {
    cmd.arg("--pad");
}

fn run_cat<X, Y, Z, F>(test_name: &str, which: &str, rows1: X, rows2: Y,
                       modify_cmd: F) -> Z
          where X: Csv, Y: Csv, Z: Csv, F: FnOnce(&mut process::Command) {
    let wrk = Workdir::new(test_name);
    wrk.create("in1.csv", rows1);
    wrk.create("in2.csv", rows2);

    let mut cmd = wrk.command("cat");
    modify_cmd(cmd.arg(which).arg("in1.csv").arg("in2.csv"));
    wrk.read_stdout(&mut cmd)
}

#[test]
fn prop_cat_rows() {
    fn p(rows: CsvData) -> bool {
        let expected = rows.clone();
        let (rows1, rows2) =
            if rows.is_empty() {
                (vec![], vec![])
            } else {
                let (rows1, rows2) = rows.split_at(rows.len() / 2);
                (rows1.to_vec(), rows2.to_vec())
            };
        let got: CsvData = run_cat("cat_rows", "rows",
                                   rows1, rows2, no_headers);
        rassert_eq!(got, expected)
    }
    qcheck(p as fn(CsvData) -> bool);
}

#[test]
fn cat_rows_space() {
    let rows = vec![svec!["\u{0085}"]];
    let expected = rows.clone();
    let (rows1, rows2) =
        if rows.is_empty() {
            (vec![], vec![])
        } else {
            let (rows1, rows2) = rows.split_at(rows.len() / 2);
            (rows1.to_vec(), rows2.to_vec())
        };
    let got: Vec<Vec<String>> =
        run_cat("cat_rows_space", "rows", rows1, rows2, no_headers);
    assert_eq!(got, expected);
}

#[test]
fn cat_rows_headers() {
    let rows1 = vec![svec!["h1", "h2"], svec!["a", "b"]];
    let rows2 = vec![svec!["h1", "h2"], svec!["y", "z"]];

    let mut expected = rows1.clone();
    expected.extend(rows2.clone().into_iter().skip(1));

    let got: Vec<Vec<String>> = run_cat("cat_rows_headers", "rows",
                                        rows1, rows2, |_| ());
    assert_eq!(got, expected);
}

#[test]
fn prop_cat_cols() {
    fn p(rows1: CsvData, rows2: CsvData) -> bool {
        let got: Vec<Vec<String>> = run_cat(
            "cat_cols", "columns", rows1.clone(), rows2.clone(), no_headers);

        let mut expected: Vec<Vec<String>> = vec![];
        let (rows1, rows2) = (rows1.to_vecs().into_iter(),
                              rows2.to_vecs().into_iter());
        for (mut r1, r2) in rows1.zip(rows2) {
            r1.extend(r2.into_iter());
            expected.push(r1);
        }
        rassert_eq!(got, expected)
    }
    qcheck(p as fn(CsvData, CsvData) -> bool);
}

#[test]
fn cat_cols_headers() {
    let rows1 = vec![svec!["h1", "h2"], svec!["a", "b"]];
    let rows2 = vec![svec!["h3", "h4"], svec!["y", "z"]];

    let expected = vec![
        svec!["h1", "h2", "h3", "h4"],
        svec!["a", "b", "y", "z"],
    ];
    let got: Vec<Vec<String>> = run_cat("cat_cols_headers", "columns",
                                        rows1, rows2, |_| ());
    assert_eq!(got, expected);
}

#[test]
fn cat_cols_no_pad() {
    let rows1 = vec![svec!["a", "b"]];
    let rows2 = vec![svec!["y", "z"], svec!["y", "z"]];

    let expected = vec![
        svec!["a", "b", "y", "z"],
    ];
    let got: Vec<Vec<String>> = run_cat("cat_cols_headers", "columns",
                                        rows1, rows2, no_headers);
    assert_eq!(got, expected);
}

#[test]
fn cat_cols_pad() {
    let rows1 = vec![svec!["a", "b"]];
    let rows2 = vec![svec!["y", "z"], svec!["y", "z"]];

    let expected = vec![
        svec!["a", "b", "y", "z"],
        svec!["", "", "y", "z"],
    ];
    let got: Vec<Vec<String>> = run_cat("cat_cols_headers", "columns",
                                        rows1, rows2, pad);
    assert_eq!(got, expected);
}

--- MODULE SEPARATOR ---
use {CsvData, qcheck};
use workdir::Workdir;

/// This tests whether `xsv count` gets the right answer.
///
/// It does some simple case analysis to handle whether we want to test counts
/// in the presence of headers and/or indexes.
fn prop_count_len(name: &str, rows: CsvData,
                  headers: bool, idx: bool) -> bool {
    let mut expected_count = rows.len();
    if headers && expected_count > 0 {
        expected_count -= 1;
    }

    let wrk = Workdir::new(name);
    if idx {
        wrk.create_indexed("in.csv", rows);
    } else {
        wrk.create("in.csv", rows);
    }

    let mut cmd = wrk.command("count");
    if !headers {
        cmd.arg("--no-headers");
    }
    cmd.arg("in.csv");

    let got_count: usize = wrk.stdout(&mut cmd);
    rassert_eq!(got_count, expected_count)
}

#[test]
fn prop_count() {
    fn p(rows: CsvData) -> bool {
        prop_count_len("prop_count", rows, false, false)
    }
    qcheck(p as fn(CsvData) -> bool);
}

#[test]
fn prop_count_headers() {
    fn p(rows: CsvData) -> bool {
        prop_count_len("prop_count_headers", rows, true, false)
    }
    qcheck(p as fn(CsvData) -> bool);
}

#[test]
fn prop_count_indexed() {
    fn p(rows: CsvData) -> bool {
        prop_count_len("prop_count_indexed", rows, false, true)
    }
    qcheck(p as fn(CsvData) -> bool);
}

#[test]
fn prop_count_indexed_headers() {
    fn p(rows: CsvData) -> bool {
        prop_count_len("prop_count_indexed_headers", rows, true, true)
    }
    qcheck(p as fn(CsvData) -> bool);
}

--- MODULE SEPARATOR ---
use quickcheck::TestResult;

use {CsvRecord, qcheck};
use workdir::Workdir;

fn trim_trailing_empty(it : &CsvRecord) -> Vec<String> {
    let mut cloned = it.clone().unwrap();
    while cloned.len() > 1 && cloned.last().unwrap().is_empty() {
        cloned.pop();
    }
    cloned
}

#[test]
fn prop_fixlengths_all_maxlen() {
    fn p(rows: Vec<CsvRecord>) -> TestResult {
        let expected_len =
            match rows.iter().map(|r| trim_trailing_empty(r).len()).max() {
                None => return TestResult::discard(),
                Some(n) => n,
            };

        let wrk = Workdir::new("fixlengths_all_maxlen").flexible(true);
        wrk.create("in.csv", rows);

        let mut cmd = wrk.command("fixlengths");
        cmd.arg("in.csv");

        let got: Vec<CsvRecord> = wrk.read_stdout(&mut cmd);
        let got_len = got.iter().map(|r| r.len()).max().unwrap();
        for r in got.iter() { assert_eq!(r.len(), got_len) }
        TestResult::from_bool(rassert_eq!(got_len, expected_len))
    }
    qcheck(p as fn(Vec<CsvRecord>) -> TestResult);
}

#[test]
fn fixlengths_all_maxlen_trims() {
    let rows = vec![
        svec!["h1", "h2"],
        svec!["abcdef", "ghijkl", "", ""],
        svec!["mnopqr", "stuvwx", "", ""],
    ];

    let wrk = Workdir::new("fixlengths_all_maxlen_trims").flexible(true);
    wrk.create("in.csv", rows);

    let mut cmd = wrk.command("fixlengths");
    cmd.arg("in.csv");

    let got: Vec<CsvRecord> = wrk.read_stdout(&mut cmd);
    for r in got.iter() { assert_eq!(r.len(), 2) }
}

#[test]
fn fixlengths_all_maxlen_trims_at_least_1() {
    let rows = vec![
        svec![""],
        svec!["", ""],
        svec!["", "", ""],
    ];

    let wrk = Workdir::new("fixlengths_all_maxlen_trims_at_least_1").flexible(true);
    wrk.create("in.csv", rows);

    let mut cmd = wrk.command("fixlengths");
    cmd.arg("in.csv");

    let got: Vec<CsvRecord> = wrk.read_stdout(&mut cmd);
    for r in got.iter() { assert_eq!(r.len(), 1) }
}


#[test]
fn prop_fixlengths_explicit_len() {
    fn p(rows: Vec<CsvRecord>, expected_len: usize) -> TestResult {
        if expected_len == 0 || rows.is_empty() {
            return TestResult::discard();
        }

        let wrk = Workdir::new("fixlengths_explicit_len").flexible(true);
        wrk.create("in.csv", rows);

        let mut cmd = wrk.command("fixlengths");
        cmd.arg("in.csv").args(&["-l", &*expected_len.to_string()]);

        let got: Vec<CsvRecord> = wrk.read_stdout(&mut cmd);
        let got_len = got.iter().map(|r| r.len()).max().unwrap();
        for r in got.iter() { assert_eq!(r.len(), got_len) }
        TestResult::from_bool(rassert_eq!(got_len, expected_len))
    }
    qcheck(p as fn(Vec<CsvRecord>, usize) -> TestResult);
}

--- MODULE SEPARATOR ---
use std::process;

use workdir::Workdir;

fn setup(name: &str) -> (Workdir, process::Command) {
    let rows = vec![
        svec!["h1", "h2"],
        svec!["abcdef", "ghijkl"],
        svec!["mnopqr", "stuvwx"],
    ];

    let wrk = Workdir::new(name);
    wrk.create("in.csv", rows);

    let mut cmd = wrk.command("flatten");
    cmd.arg("in.csv");

    (wrk, cmd)
}

#[test]
fn flatten_basic() {
    let (wrk, mut cmd) = setup("flatten_basic");
    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1  abcdef
h2  ghijkl
#
h1  mnopqr
h2  stuvwx\
";
    assert_eq!(got, expected.to_string());
}

#[test]
fn flatten_no_headers() {
    let (wrk, mut cmd) = setup("flatten_no_headers");
    cmd.arg("--no-headers");

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
0   h1
1   h2
#
0   abcdef
1   ghijkl
#
0   mnopqr
1   stuvwx\
";
    assert_eq!(got, expected.to_string());
}

#[test]
fn flatten_separator() {
    let (wrk, mut cmd) = setup("flatten_separator");
    cmd.args(&["--separator", "!mysep!"]);

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1  abcdef
h2  ghijkl
!mysep!
h1  mnopqr
h2  stuvwx\
";
    assert_eq!(got, expected.to_string());
}

#[test]
fn flatten_condense() {
    let (wrk, mut cmd) = setup("flatten_condense");
    cmd.args(&["--condense", "2"]);

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1  ab...
h2  gh...
#
h1  mn...
h2  st...\
";
    assert_eq!(got, expected.to_string());
}

--- MODULE SEPARATOR ---
use std::process;

use workdir::Workdir;

fn setup(name: &str) -> (Workdir, process::Command) {
    let rows = vec![
        svec!["h1", "h2"],
        svec!["abcdef", "ghijkl"],
        svec!["mnopqr", "stuvwx"],
    ];

    let wrk = Workdir::new(name);
    wrk.create("in.csv", rows);

    let mut cmd = wrk.command("fmt");
    cmd.arg("in.csv");

    (wrk, cmd)
}

#[test]
fn fmt_delimiter() {
    let (wrk, mut cmd) = setup("fmt_delimiter");
    cmd.args(&["--out-delimiter", "\t"]);

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1\th2
abcdef\tghijkl
mnopqr\tstuvwx";
    assert_eq!(got, expected.to_string());
}

#[test]
fn fmt_weird_delimiter() {
    let (wrk, mut cmd) = setup("fmt_weird_delimiter");
    cmd.args(&["--out-delimiter", "h"]);

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
\"h1\"h\"h2\"
abcdefh\"ghijkl\"
mnopqrhstuvwx";
    assert_eq!(got, expected.to_string());
}

#[test]
fn fmt_crlf() {
    let (wrk, mut cmd) = setup("fmt_crlf");
    cmd.arg("--crlf");

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1,h2\r
abcdef,ghijkl\r
mnopqr,stuvwx";
    assert_eq!(got, expected.to_string());
}

#[test]
fn fmt_quote_always() {
    let (wrk, mut cmd) = setup("fmt_quote_always");
    cmd.arg("--quote-always");

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
\"h1\",\"h2\"
\"abcdef\",\"ghijkl\"
\"mnopqr\",\"stuvwx\"";
    assert_eq!(got, expected.to_string());
}

--- MODULE SEPARATOR ---
use std::borrow::ToOwned;
use std::collections::hash_map::{HashMap, Entry};
use std::process;

use csv;
use stats::Frequencies;

use {Csv, CsvData, qcheck_sized};
use workdir::Workdir;

fn setup(name: &str) -> (Workdir, process::Command) {
    let rows = vec![
        svec!["h1", "h2"],
        svec!["a", "z"],
        svec!["a", "y"],
        svec!["a", "y"],
        svec!["b", "z"],
        svec!["", "z"],
        svec!["(NULL)", "x"],
    ];

    let wrk = Workdir::new(name);
    wrk.create("in.csv", rows);

    let mut cmd = wrk.command("frequency");
    cmd.arg("in.csv");

    (wrk, cmd)
}

#[test]
fn frequency_no_headers() {
    let (wrk, mut cmd) = setup("frequency_no_headers");
    cmd.args(&["--limit", "0"]).args(&["--select", "1"]).arg("--no-headers");

    let mut got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    got = got.into_iter().skip(1).collect();
    got.sort();
    let expected = vec![
        svec!["1", "(NULL)", "1"],
        svec!["1", "(NULL)", "1"],
        svec!["1", "a", "3"],
        svec!["1", "b", "1"],
        svec!["1", "h1", "1"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn frequency_no_nulls() {
    let (wrk, mut cmd) = setup("frequency_no_nulls");
    cmd.arg("--no-nulls").args(&["--limit", "0"]).args(&["--select", "h1"]);

    let mut got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    got.sort();
    let expected = vec![
        svec!["field", "value", "count"],
        svec!["h1", "(NULL)", "1"],
        svec!["h1", "a", "3"],
        svec!["h1", "b", "1"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn frequency_nulls() {
    let (wrk, mut cmd) = setup("frequency_nulls");
    cmd.args(&["--limit", "0"]).args(&["--select", "h1"]);

    let mut got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    got.sort();
    let expected = vec![
        svec!["field", "value", "count"],
        svec!["h1", "(NULL)", "1"],
        svec!["h1", "(NULL)", "1"],
        svec!["h1", "a", "3"],
        svec!["h1", "b", "1"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn frequency_limit() {
    let (wrk, mut cmd) = setup("frequency_limit");
    cmd.args(&["--limit", "1"]);

    let mut got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    got.sort();
    let expected = vec![
        svec!["field", "value", "count"],
        svec!["h1", "a", "3"],
        svec!["h2", "z", "3"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn frequency_asc() {
    let (wrk, mut cmd) = setup("frequency_asc");
    cmd.args(&["--limit", "1"]).args(&["--select", "h2"]).arg("--asc");

    let mut got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    got.sort();
    let expected = vec![
        svec!["field", "value", "count"],
        svec!["h2", "x", "1"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn frequency_select() {
    let (wrk, mut cmd) = setup("frequency_select");
    cmd.args(&["--limit", "0"]).args(&["--select", "h2"]);

    let mut got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    got.sort();
    let expected = vec![
        svec!["field", "value", "count"],
        svec!["h2", "x", "1"],
        svec!["h2", "y", "2"],
        svec!["h2", "z", "3"],
    ];
    assert_eq!(got, expected);
}

// This tests that a frequency table computed by `xsv` is always the same
// as the frequency table computed in memory.
#[test]
fn prop_frequency() {
    fn p(rows: CsvData) -> bool {
        param_prop_frequency("prop_frequency", rows, false)
    }
    // Run on really small values because we are incredibly careless
    // with allocation.
    qcheck_sized(p as fn(CsvData) -> bool, 2);
}


// This tests that running the frequency command on a CSV file with these two
// rows does not burst in flames:
//
//     \u{FEFF}
//     ""
//
// In this case, the `param_prop_frequency` just ignores this particular test.
// Namely, \u{FEFF} is the UTF-8 BOM, which is ignored by the underlying CSV
// reader.
#[test]
fn frequency_bom() {
    let rows = CsvData {
        data: vec![
            ::CsvRecord(vec!["\u{FEFF}".to_string()]),
            ::CsvRecord(vec!["".to_string()]),
        ],
    };
    assert!(param_prop_frequency("prop_frequency", rows, false))
}

// This tests that a frequency table computed by `xsv` (with an index) is
// always the same as the frequency table computed in memory.
#[test]
fn prop_frequency_indexed() {
    fn p(rows: CsvData) -> bool {
        param_prop_frequency("prop_frequency_indxed", rows, true)
    }
    // Run on really small values because we are incredibly careless
    // with allocation.
    qcheck_sized(p as fn(CsvData) -> bool, 2);
}

fn param_prop_frequency(name: &str, rows: CsvData, idx: bool) -> bool {
    if !rows.is_empty() && rows[0][0].len() == 3 && rows[0][0] == "\u{FEFF}" {
        return true;
    }
    let wrk = Workdir::new(name);
    if idx {
        wrk.create_indexed("in.csv", rows.clone());
    } else {
        wrk.create("in.csv", rows.clone());
    }

    let mut cmd = wrk.command("frequency");
    cmd.arg("in.csv").args(&["-j", "4"]).args(&["--limit", "0"]);

    let stdout = wrk.stdout::<String>(&mut cmd);
    let got_ftables = ftables_from_csv_string(stdout);
    let expected_ftables = ftables_from_rows(rows);
    assert_eq_ftables(&got_ftables, &expected_ftables)
}

type FTables = HashMap<String, Frequencies<String>>;

#[derive(Deserialize)]
struct FRow {
    field: String,
    value: String,
    count: usize,
}

fn ftables_from_rows<T: Csv>(rows: T) -> FTables {
    let mut rows = rows.to_vecs();
    if rows.len() <= 1 {
        return HashMap::new();
    }

    let header = rows.remove(0);
    let mut ftables = HashMap::new();
    for field in header.iter() {
        ftables.insert(field.clone(), Frequencies::new());
    }
    for row in rows.into_iter() {
        for (i, mut field) in row.into_iter().enumerate() {
            field = field.trim().to_owned();
            if field.is_empty() {
                field = "(NULL)".to_owned();
            }
            ftables.get_mut(&header[i]).unwrap().add(field);
        }
    }
    ftables
}

fn ftables_from_csv_string(data: String) -> FTables {
    let mut rdr = csv::Reader::from_reader(data.as_bytes());
    let mut ftables = HashMap::new();
    for frow in rdr.deserialize() {
        let frow: FRow = frow.unwrap();
        match ftables.entry(frow.field) {
            Entry::Vacant(v) => {
                let mut ftable = Frequencies::new();
                for _ in 0..frow.count {
                    ftable.add(frow.value.clone());
                }
                v.insert(ftable);
            }
            Entry::Occupied(mut v) => {
                for _ in 0..frow.count {
                    v.get_mut().add(frow.value.clone());
                }
            }
        }
    }
    ftables
}

fn freq_data<T>(ftable: &Frequencies<T>) -> Vec<(&T, u64)>
        where T: ::std::hash::Hash + Ord + Clone {
    let mut freqs = ftable.most_frequent();
    freqs.sort();
    freqs
}

fn assert_eq_ftables(got: &FTables, expected: &FTables) -> bool {
    for (k, v) in got.iter() {
        assert_eq!(freq_data(v), freq_data(expected.get(k).unwrap()));
    }
    for (k, v) in expected.iter() {
        assert_eq!(freq_data(got.get(k).unwrap()), freq_data(v));
    }
    true
}

--- MODULE SEPARATOR ---
use std::process;

use workdir::Workdir;

fn setup(name: &str) -> (Workdir, process::Command) {
    let rows1 = vec![svec!["h1", "h2"], svec!["a", "b"]];
    let rows2 = vec![svec!["h2", "h3"], svec!["y", "z"]];

    let wrk = Workdir::new(name);
    wrk.create("in1.csv", rows1);
    wrk.create("in2.csv", rows2);

    let mut cmd = wrk.command("headers");
    cmd.arg("in1.csv");

    (wrk, cmd)
}

#[test]
fn headers_basic() {
    let (wrk, mut cmd) = setup("headers_basic");

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
1   h1
2   h2";
    assert_eq!(got, expected.to_string());
}

#[test]
fn headers_just_names() {
    let (wrk, mut cmd) = setup("headers_just_names");
    cmd.arg("--just-names");

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1
h2";
    assert_eq!(got, expected.to_string());
}

#[test]
fn headers_multiple() {
    let (wrk, mut cmd) = setup("headers_multiple");
    cmd.arg("in2.csv");

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1
h2
h2
h3";
    assert_eq!(got, expected.to_string());
}

#[test]
fn headers_intersect() {
    let (wrk, mut cmd) = setup("headers_intersect");
    cmd.arg("in2.csv").arg("--intersect");

    let got: String = wrk.stdout(&mut cmd);
    let expected = "\
h1
h2
h3";
    assert_eq!(got, expected.to_string());
}

--- MODULE SEPARATOR ---
use std::fs;

use filetime::{FileTime, set_file_times};

use workdir::Workdir;

#[test]
fn index_outdated() {
    let wrk = Workdir::new("index_outdated");
    wrk.create_indexed("in.csv", vec![svec![""]]);

    let md = fs::metadata(&wrk.path("in.csv.idx")).unwrap();
    set_file_times(
        &wrk.path("in.csv"),
        future_time(FileTime::from_last_modification_time(&md)),
        future_time(FileTime::from_last_access_time(&md)),
    ).unwrap();

    let mut cmd = wrk.command("count");
    cmd.arg("--no-headers").arg("in.csv");
    wrk.assert_err(&mut cmd);
}

fn future_time(ft: FileTime) -> FileTime {
    let secs = ft.seconds_relative_to_1970();
    FileTime::from_seconds_since_1970(secs + 10_000, 0)
}

--- MODULE SEPARATOR ---
use workdir::Workdir;

// This macro takes *two* identifiers: one for the test with headers
// and another for the test without headers.
macro_rules! join_test {
    ($name:ident, $fun:expr) => (
        mod $name {
            use std::process;

            use workdir::Workdir;
            use super::{make_rows, setup};

            #[test]
            fn headers() {
                let wrk = setup(stringify!($name), true);
                let mut cmd = wrk.command("join");
                cmd.args(&["city", "cities.csv", "city", "places.csv"]);
                $fun(wrk, cmd, true);
            }

            #[test]
            fn no_headers() {
                let n = stringify!(concat_idents!($name, _no_headers));
                let wrk = setup(n, false);
                let mut cmd = wrk.command("join");
                cmd.arg("--no-headers");
                cmd.args(&["1", "cities.csv", "1", "places.csv"]);
                $fun(wrk, cmd, false);
            }
        }
    );
}

fn setup(name: &str, headers: bool) -> Workdir {
    let mut cities = vec![
        svec!["Boston", "MA"],
        svec!["New York", "NY"],
        svec!["San Francisco", "CA"],
        svec!["Buffalo", "NY"],
    ];
    let mut places = vec![
        svec!["Boston", "Logan Airport"],
        svec!["Boston", "Boston Garden"],
        svec!["Buffalo", "Ralph Wilson Stadium"],
        svec!["Orlando", "Disney World"],
    ];
    if headers { cities.insert(0, svec!["city", "state"]); }
    if headers { places.insert(0, svec!["city", "place"]); }

    let wrk = Workdir::new(name);
    wrk.create("cities.csv", cities);
    wrk.create("places.csv", places);
    wrk
}

fn make_rows(headers: bool, rows: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let mut all_rows = vec![];
    if headers {
        all_rows.push(svec!["city", "state", "city", "place"]);
    }
    all_rows.extend(rows.into_iter());
    all_rows
}

join_test!(join_inner,
           |wrk: Workdir, mut cmd: process::Command, headers: bool| {
    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = make_rows(headers, vec![
        svec!["Boston", "MA", "Boston", "Logan Airport"],
        svec!["Boston", "MA", "Boston", "Boston Garden"],
        svec!["Buffalo", "NY", "Buffalo", "Ralph Wilson Stadium"],
    ]);
    assert_eq!(got, expected);
});

join_test!(join_outer_left,
           |wrk: Workdir, mut cmd: process::Command, headers: bool| {
    cmd.arg("--left");
    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = make_rows(headers, vec![
        svec!["Boston", "MA", "Boston", "Logan Airport"],
        svec!["Boston", "MA", "Boston", "Boston Garden"],
        svec!["New York", "NY", "", ""],
        svec!["San Francisco", "CA", "", ""],
        svec!["Buffalo", "NY", "Buffalo", "Ralph Wilson Stadium"],
    ]);
    assert_eq!(got, expected);
});

join_test!(join_outer_right,
           |wrk: Workdir, mut cmd: process::Command, headers: bool| {
    cmd.arg("--right");
    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = make_rows(headers, vec![
        svec!["Boston", "MA", "Boston", "Logan Airport"],
        svec!["Boston", "MA", "Boston", "Boston Garden"],
        svec!["Buffalo", "NY", "Buffalo", "Ralph Wilson Stadium"],
        svec!["", "", "Orlando", "Disney World"],
    ]);
    assert_eq!(got, expected);
});

join_test!(join_outer_full,
           |wrk: Workdir, mut cmd: process::Command, headers: bool| {
    cmd.arg("--full");
    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = make_rows(headers, vec![
        svec!["Boston", "MA", "Boston", "Logan Airport"],
        svec!["Boston", "MA", "Boston", "Boston Garden"],
        svec!["New York", "NY", "", ""],
        svec!["San Francisco", "CA", "", ""],
        svec!["Buffalo", "NY", "Buffalo", "Ralph Wilson Stadium"],
        svec!["", "", "Orlando", "Disney World"],
    ]);
    assert_eq!(got, expected);
});

#[test]
fn join_inner_issue11() {
    let a = vec![
        svec!["1", "2"],
        svec!["3", "4"],
        svec!["5", "6"],
    ];
    let b = vec![
        svec!["2", "1"],
        svec!["4", "3"],
        svec!["6", "5"],
    ];

    let wrk = Workdir::new("join_inner_issue11");
    wrk.create("a.csv", a);
    wrk.create("b.csv", b);

    let mut cmd = wrk.command("join");
    cmd.args(&["1,2", "a.csv", "2,1", "b.csv"]);

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["1", "2", "2", "1"],
        svec!["3", "4", "4", "3"],
        svec!["5", "6", "6", "5"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn join_cross() {
    let wrk = Workdir::new("join_cross");
    wrk.create("letters.csv",
               vec![svec!["h1", "h2"], svec!["a", "b"], svec!["c", "d"]]);
    wrk.create("numbers.csv",
               vec![svec!["h3", "h4"], svec!["1", "2"], svec!["3", "4"]]);

    let mut cmd = wrk.command("join");
    cmd.arg("--cross")
       .args(&["", "letters.csv", "", "numbers.csv"]);
    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["h1", "h2", "h3", "h4"],
        svec!["a", "b", "1", "2"],
        svec!["a", "b", "3", "4"],
        svec!["c", "d", "1", "2"],
        svec!["c", "d", "3", "4"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn join_cross_no_headers() {
    let wrk = Workdir::new("join_cross_no_headers");
    wrk.create("letters.csv", vec![svec!["a", "b"], svec!["c", "d"]]);
    wrk.create("numbers.csv", vec![svec!["1", "2"], svec!["3", "4"]]);

    let mut cmd = wrk.command("join");
    cmd.arg("--cross").arg("--no-headers")
       .args(&["", "letters.csv", "", "numbers.csv"]);
    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["a", "b", "1", "2"],
        svec!["a", "b", "3", "4"],
        svec!["c", "d", "1", "2"],
        svec!["c", "d", "3", "4"],
    ];
    assert_eq!(got, expected);
}

--- MODULE SEPARATOR ---
use std::borrow::ToOwned;

use workdir::Workdir;

macro_rules! part_eq {
    ($wrk:expr, $path:expr, $expected:expr) => (
        assert_eq!($wrk.from_str::<String>(&$wrk.path($path)),
                   $expected.to_owned());
    );
}

fn data(headers: bool) -> Vec<Vec<String>> {
    let mut rows = vec![
        svec!["NY", "Manhatten"],
        svec!["CA", "San Francisco"],
        svec!["TX", "Dallas"],
        svec!["NY", "Buffalo"],
        svec!["TX", "Fort Worth"],
    ];
    if headers { rows.insert(0, svec!["state", "city"]); }
    rows
}

#[test]
fn partition() {
    let wrk = Workdir::new("partition");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("partition");
    cmd.arg("state").arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    part_eq!(wrk, "CA.csv", "\
state,city
CA,San Francisco
");
    part_eq!(wrk, "NY.csv", "\
state,city
NY,Manhatten
NY,Buffalo
");
    part_eq!(wrk, "TX.csv", "\
state,city
TX,Dallas
TX,Fort Worth
");
}

#[test]
fn partition_drop() {
    let wrk = Workdir::new("partition");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("partition");
    cmd.arg("--drop").arg("state").arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    part_eq!(wrk, "CA.csv", "\
city
San Francisco
");
    part_eq!(wrk, "NY.csv", "\
city
Manhatten
Buffalo
");
    part_eq!(wrk, "TX.csv", "\
city
Dallas
Fort Worth
");
}

#[test]
fn partition_without_headers() {
    let wrk = Workdir::new("partition_without_headers");
    wrk.create("in.csv", data(false));

    let mut cmd = wrk.command("partition");
    cmd.arg("--no-headers").arg("1").arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    part_eq!(wrk, "CA.csv", "\
CA,San Francisco
");
    part_eq!(wrk, "NY.csv", "\
NY,Manhatten
NY,Buffalo
");
    part_eq!(wrk, "TX.csv", "\
TX,Dallas
TX,Fort Worth
");
}

#[test]
fn partition_drop_without_headers() {
    let wrk = Workdir::new("partition_without_headers");
    wrk.create("in.csv", data(false));

    let mut cmd = wrk.command("partition");
    cmd.arg("--drop").arg("--no-headers").arg("1").arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    part_eq!(wrk, "CA.csv", "\
San Francisco
");
    part_eq!(wrk, "NY.csv", "\
Manhatten
Buffalo
");
    part_eq!(wrk, "TX.csv", "\
Dallas
Fort Worth
");
}

#[test]
fn partition_into_new_directory() {
    let wrk = Workdir::new("partition_into_new_directory");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("partition");
    cmd.arg("state").arg(&wrk.path("out")).arg("in.csv");
    wrk.run(&mut cmd);

    assert!(wrk.path("out/NY.csv").exists());
}

#[test]
fn partition_custom_filename() {
    let wrk = Workdir::new("partition_custom_filename");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("partition");
    cmd.args(&["--filename", "state-{}-partition.csv"])
        .arg("state")
        .arg(&wrk.path("."))
        .arg("in.csv");
    wrk.run(&mut cmd);

    assert!(wrk.path("state-NY-partition.csv").exists());
}

#[test]
fn partition_custom_filename_with_directory() {
    let wrk = Workdir::new("partition_custom_filename_with_directory");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("partition");
    cmd.args(&["--filename", "{}/cities.csv"])
        .arg("state")
        .arg(&wrk.path("."))
        .arg("in.csv");
    wrk.run(&mut cmd);

    // This variation also helps with parallel partition jobs.
    assert!(wrk.path("NY/cities.csv").exists());
}

#[test]
fn partition_invalid_filename() {
    let wrk = Workdir::new("partition_invalid_filename");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("partition");
    cmd.args(&["--filename", "foo.csv"])
        .arg("state")
        .arg(&wrk.path("."))
        .arg("in.csv");
    wrk.assert_err(&mut cmd);

    let mut cmd = wrk.command("partition");
    cmd.args(&["--filename", "{}{}.csv"])
        .arg("state")
        .arg(&wrk.path("."))
        .arg("in.csv");
    wrk.assert_err(&mut cmd);
}

fn tricky_data() -> Vec<Vec<String>> {
    vec![
        svec!["key", "explanation"],
        svec!["", "empty key"],
        svec!["empty", "the string empty"],
        svec!["unsafe _1$!,\"", "unsafe in shell"],
        svec!["collision", "ordinary value"],
        svec!["collision", "in same file"],
        svec!["coll ision", "collides"],
        svec!["collision!", "collides again"],
        svec!["collision_2", "collides with disambiguated"],
    ]
}

#[test]
fn partition_with_tricky_key_values() {
    let wrk = Workdir::new("partition_with_tricky_key_values");
    wrk.create("in.csv", tricky_data());

    let mut cmd = wrk.command("partition");
    cmd.arg("key").arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    part_eq!(wrk, "empty.csv", "\
key,explanation
,empty key
");
     part_eq!(wrk, "empty_1.csv", "\
key,explanation
empty,the string empty
");
     part_eq!(wrk, "unsafe_1.csv", r#"key,explanation
"unsafe _1$!,""",unsafe in shell
"#);
     part_eq!(wrk, "collision.csv", "\
key,explanation
collision,ordinary value
collision,in same file
");
    part_eq!(wrk, "collision_2.csv", "\
key,explanation
coll ision,collides
");
    part_eq!(wrk, "collision_3.csv", "\
key,explanation
collision!,collides again
");
    // Tricky! We didn't see this an input, but we did generate it as an
    // output already.
    part_eq!(wrk, "collision_2_4.csv", "\
key,explanation
collision_2,collides with disambiguated
");
}

fn prefix_data() -> Vec<Vec<String>> {
    vec![
        svec!["state", "city"],
        svec!["MA", "Boston"],
        svec!["ME", "Portland"],
        svec!["M", "Too short"],
        svec!["CA", "San Francisco"],
        svec!["CO", "Denver"],
    ]
}

#[test]
fn partition_with_prefix_length() {
    let wrk = Workdir::new("partition_with_prefix_length");
    wrk.create("in.csv", prefix_data());

    let mut cmd = wrk.command("partition");
    cmd
        .args(&["--prefix-length", "1"])
        .arg("state")
        .arg(&wrk.path("."))
        .arg("in.csv");
    wrk.run(&mut cmd);

    part_eq!(wrk, "M.csv", "\
state,city
MA,Boston
ME,Portland
M,Too short
");
    part_eq!(wrk, "C.csv", "\
state,city
CA,San Francisco
CO,Denver
");
}

--- MODULE SEPARATOR ---
use workdir::Workdir;

use {Csv, CsvData, qcheck};

fn prop_reverse(name: &str, rows: CsvData, headers: bool) -> bool {
    let wrk = Workdir::new(name);
    wrk.create("in.csv", rows.clone());

    let mut cmd = wrk.command("reverse");
    cmd.arg("in.csv");
    if !headers { cmd.arg("--no-headers"); }

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let mut expected = rows.to_vecs();
    let headers = if headers && !expected.is_empty() {
        expected.remove(0)
    } else {
        vec![]
    };
    expected.reverse();
    if !headers.is_empty() { expected.insert(0, headers); }
    rassert_eq!(got, expected)
}

#[test]
fn prop_reverse_headers() {
    fn p(rows: CsvData) -> bool {
        prop_reverse("prop_reverse_headers", rows, true)
    }
    qcheck(p as fn(CsvData) -> bool);
}

#[test]
fn prop_reverse_no_headers() {
    fn p(rows: CsvData) -> bool {
        prop_reverse("prop_reverse_no_headers", rows, false)
    }
    qcheck(p as fn(CsvData) -> bool);
}

--- MODULE SEPARATOR ---
use workdir::Workdir;

fn data(headers: bool) -> Vec<Vec<String>> {
    let mut rows = vec![
        svec!["foobar", "barfoo"],
        svec!["a", "b"],
        svec!["barfoo", "foobar"],
    ];
    if headers { rows.insert(0, svec!["h1", "h2"]); }
    rows
}

#[test]
fn search() {
    let wrk = Workdir::new("search");
    wrk.create("data.csv", data(true));
    let mut cmd = wrk.command("search");
    cmd.arg("^foo").arg("data.csv");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["h1", "h2"],
        svec!["foobar", "barfoo"],
        svec!["barfoo", "foobar"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn search_empty() {
    let wrk = Workdir::new("search");
    wrk.create("data.csv", data(true));
    let mut cmd = wrk.command("search");
    cmd.arg("xxx").arg("data.csv");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["h1", "h2"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn search_empty_no_headers() {
    let wrk = Workdir::new("search");
    wrk.create("data.csv", data(true));
    let mut cmd = wrk.command("search");
    cmd.arg("xxx").arg("data.csv");
    cmd.arg("--no-headers");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected: Vec<Vec<String>> = vec![];
    assert_eq!(got, expected);
}

#[test]
fn search_ignore_case() {
    let wrk = Workdir::new("search");
    wrk.create("data.csv", data(true));
    let mut cmd = wrk.command("search");
    cmd.arg("^FoO").arg("data.csv");
    cmd.arg("--ignore-case");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["h1", "h2"],
        svec!["foobar", "barfoo"],
        svec!["barfoo", "foobar"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn search_no_headers() {
    let wrk = Workdir::new("search_no_headers");
    wrk.create("data.csv", data(false));
    let mut cmd = wrk.command("search");
    cmd.arg("^foo").arg("data.csv");
    cmd.arg("--no-headers");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["foobar", "barfoo"],
        svec!["barfoo", "foobar"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn search_select() {
    let wrk = Workdir::new("search_select");
    wrk.create("data.csv", data(true));
    let mut cmd = wrk.command("search");
    cmd.arg("^foo").arg("data.csv");
    cmd.arg("--select").arg("h2");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["h1", "h2"],
        svec!["barfoo", "foobar"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn search_select_no_headers() {
    let wrk = Workdir::new("search_select_no_headers");
    wrk.create("data.csv", data(false));
    let mut cmd = wrk.command("search");
    cmd.arg("^foo").arg("data.csv");
    cmd.arg("--select").arg("2");
    cmd.arg("--no-headers");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["barfoo", "foobar"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn search_invert_match() {
    let wrk = Workdir::new("search_invert_match");
    wrk.create("data.csv", data(false));
    let mut cmd = wrk.command("search");
    cmd.arg("^foo").arg("data.csv");
    cmd.arg("--invert-match");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["foobar", "barfoo"],
        svec!["a", "b"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn search_invert_match_no_headers() {
    let wrk = Workdir::new("search_invert_match");
    wrk.create("data.csv", data(false));
    let mut cmd = wrk.command("search");
    cmd.arg("^foo").arg("data.csv");
    cmd.arg("--invert-match");
    cmd.arg("--no-headers");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["a", "b"],
    ];
    assert_eq!(got, expected);
}

--- MODULE SEPARATOR ---
use workdir::Workdir;

macro_rules! select_test {
    ($name:ident, $select:expr, $select_no_headers:expr,
     $expected_headers:expr, $expected_rows:expr) => (
        mod $name {
            use workdir::Workdir;
            use super::data;

            #[test]
            fn headers() {
                let wrk = Workdir::new(stringify!($name));
                wrk.create("data.csv", data(true));
                let mut cmd = wrk.command("select");
                cmd.arg("--").arg($select).arg("data.csv");
                let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);

                let expected = vec![
                    $expected_headers.iter()
                                     .map(|s| s.to_string())
                                     .collect::<Vec<String>>(),
                    $expected_rows.iter()
                                  .map(|s| s.to_string())
                                  .collect::<Vec<String>>(),
                ];
                assert_eq!(got, expected);
            }

            #[test]
            fn no_headers() {
                let wrk = Workdir::new(stringify!($name));
                wrk.create("data.csv", data(false));
                let mut cmd = wrk.command("select");
                cmd.arg("--no-headers")
                   .arg("--").arg($select_no_headers).arg("data.csv");
                let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);

                let expected = vec![
                    $expected_rows.iter()
                                  .map(|s| s.to_string())
                                  .collect::<Vec<String>>(),
                ];
                assert_eq!(got, expected);
            }
        }
    );
}

macro_rules! select_test_err {
    ($name:ident, $select:expr) => (
        #[test]
        fn $name() {
            let wrk = Workdir::new(stringify!($name));
            wrk.create("data.csv", data(true));
            let mut cmd = wrk.command("select");
            cmd.arg($select).arg("data.csv");
            wrk.assert_err(&mut cmd);
        }
    );
}

fn header_row() -> Vec<String> { svec!["h1", "h2", "h[]3", "h4", "h1"] }

fn data(headers: bool) -> Vec<Vec<String>> {
    let mut rows = vec![
        svec!["a", "b", "c", "d", "e"],
    ];
    if headers { rows.insert(0, header_row()) }
    rows
}

select_test!(select_simple, "h1", "1", ["h1"], ["a"]);
select_test!(select_simple_idx, "h1[0]", "1", ["h1"], ["a"]);
select_test!(select_simple_idx_2, "h1[1]", "5", ["h1"], ["e"]);

select_test!(select_quoted, r#""h[]3""#, "3", ["h[]3"], ["c"]);
select_test!(select_quoted_idx, r#""h[]3"[0]"#, "3", ["h[]3"], ["c"]);

select_test!(select_range, "h1-h4", "1-4",
             ["h1", "h2", "h[]3", "h4"], ["a", "b", "c", "d"]);

select_test!(select_range_multi, r#"h1-h2,"h[]3"-h4"#, "1-2,3-4",
             ["h1", "h2", "h[]3", "h4"], ["a", "b", "c", "d"]);
select_test!(select_range_multi_idx, r#"h1-h2,"h[]3"[0]-h4"#, "1-2,3-4",
             ["h1", "h2", "h[]3", "h4"], ["a", "b", "c", "d"]);

select_test!(select_reverse, "h1[1]-h1[0]", "5-1",
             ["h1", "h4", "h[]3", "h2", "h1"], ["e", "d", "c", "b", "a"]);

select_test!(select_not, r#"!"h[]3"[0]"#, "!3",
             ["h1", "h2", "h4", "h1"], ["a", "b", "d", "e"]);
select_test!(select_not_range, "!h1[1]-h2", "!5-2", ["h1"], ["a"]);

select_test!(select_duplicate, "h1,h1", "1,1", ["h1", "h1"], ["a", "a"]);
select_test!(select_duplicate_range, "h1-h2,h1-h2", "1-2,1-2",
             ["h1", "h2", "h1", "h2"], ["a", "b", "a", "b"]);
select_test!(select_duplicate_range_reverse, "h1-h2,h2-h1", "1-2,2-1",
             ["h1", "h2", "h2", "h1"], ["a", "b", "b", "a"]);

select_test!(select_range_no_end, "h4-", "4-", ["h4", "h1"], ["d", "e"]);
select_test!(select_range_no_start, "-h2", "-2", ["h1", "h2"], ["a", "b"]);
select_test!(select_range_no_end_cat, "h4-,h1", "4-,1",
             ["h4", "h1", "h1"], ["d", "e", "a"]);
select_test!(select_range_no_start_cat, "-h2,h1[1]", "-2,5",
             ["h1", "h2", "h1"], ["a", "b", "e"]);

select_test_err!(select_err_unknown_header, "dne");
select_test_err!(select_err_oob_low, "0");
select_test_err!(select_err_oob_high, "6");
select_test_err!(select_err_idx_as_name, "1[0]");
select_test_err!(select_err_idx_oob_high, "h1[2]");
select_test_err!(select_err_idx_not_int, "h1[2.0]");
select_test_err!(select_err_idx_not_int_2, "h1[a]");
select_test_err!(select_err_unclosed_quote, r#""h1"#);
select_test_err!(select_err_unclosed_bracket, r#""h1"[1"#);
select_test_err!(select_err_expected_end_of_field, "a-b-");

--- MODULE SEPARATOR ---
use std::borrow::ToOwned;
use std::process;

use workdir::Workdir;

macro_rules! slice_tests {
    ($name:ident, $start:expr, $end:expr, $expected:expr) => (
        mod $name {
            use super::test_slice;

            #[test]
            fn headers_no_index() {
                let name = concat!(stringify!($name), "headers_no_index");
                test_slice(name, $start, $end, $expected, true, false, false);
            }

            #[test]
            fn no_headers_no_index() {
                let name = concat!(stringify!($name), "no_headers_no_index");
                test_slice(name, $start, $end, $expected, false, false, false);
            }

            #[test]
            fn headers_index() {
                let name = concat!(stringify!($name), "headers_index");
                test_slice(name, $start, $end, $expected, true, true, false);
            }

            #[test]
            fn no_headers_index() {
                let name = concat!(stringify!($name), "no_headers_index");
                test_slice(name, $start, $end, $expected, false, true, false);
            }

            #[test]
            fn headers_no_index_len() {
                let name = concat!(stringify!($name), "headers_no_index_len");
                test_slice(name, $start, $end, $expected, true, false, true);
            }

            #[test]
            fn no_headers_no_index_len() {
                let name = concat!(stringify!($name),
                                   "no_headers_no_index_len");
                test_slice(name, $start, $end, $expected, false, false, true);
            }

            #[test]
            fn headers_index_len() {
                let name = concat!(stringify!($name), "headers_index_len");
                test_slice(name, $start, $end, $expected, true, true, true);
            }

            #[test]
            fn no_headers_index_len() {
                let name = concat!(stringify!($name), "no_headers_index_len");
                test_slice(name, $start, $end, $expected, false, true, true);
            }
        }
    );
}

fn setup(name: &str, headers: bool, use_index: bool)
        -> (Workdir, process::Command) {
    let wrk = Workdir::new(name);
    let mut data = vec![
        svec!["a"], svec!["b"], svec!["c"], svec!["d"], svec!["e"]
    ];
    if headers { data.insert(0, svec!["header"]); }
    if use_index {
        wrk.create_indexed("in.csv", data);
    } else {
        wrk.create("in.csv", data);
    }

    let mut cmd = wrk.command("slice");
    cmd.arg("in.csv");

    (wrk, cmd)
}

fn test_slice(name: &str, start: Option<usize>, end: Option<usize>,
              expected: &[&str], headers: bool,
              use_index: bool, as_len: bool) {
    let (wrk, mut cmd) = setup(name, headers, use_index);
    if let Some(start) = start {
        cmd.arg("--start").arg(&start.to_string());
    }
    if let Some(end) = end {
        if as_len {
            let start = start.unwrap_or(0);
            cmd.arg("--len").arg(&(end - start).to_string());
        } else {
            cmd.arg("--end").arg(&end.to_string());
        }
    }
    if !headers {
        cmd.arg("--no-headers");
    }

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let mut expected = expected.iter()
                               .map(|&s| vec![s.to_owned()])
                               .collect::<Vec<Vec<String>>>();
    if headers { expected.insert(0, svec!["header"]); }
    assert_eq!(got, expected);
}

fn test_index(name: &str, idx: usize, expected: &str,
              headers: bool, use_index: bool) {
    let (wrk, mut cmd) = setup(name, headers, use_index);
    cmd.arg("--index").arg(&idx.to_string());
    if !headers {
        cmd.arg("--no-headers");
    }

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let mut expected = vec![vec![expected.to_owned()]];
    if headers { expected.insert(0, svec!["header"]); }
    assert_eq!(got, expected);
}

slice_tests!(slice_simple, Some(0), Some(1), &["a"]);
slice_tests!(slice_simple_2, Some(1), Some(3), &["b", "c"]);
slice_tests!(slice_no_start, None, Some(1), &["a"]);
slice_tests!(slice_no_end, Some(3), None, &["d", "e"]);
slice_tests!(slice_all, None, None, &["a", "b", "c", "d", "e"]);

#[test]
fn slice_index() {
    test_index("slice_index", 1, "b", true, false);
}
#[test]
fn slice_index_no_headers() {
    test_index("slice_index_no_headers", 1, "b", false, false);
}
#[test]
fn slice_index_withindex() {
    test_index("slice_index_withindex", 1, "b", true, true);
}
#[test]
fn slice_index_no_headers_withindex() {
    test_index("slice_index_no_headers_withindex", 1, "b", false, true);
}

--- MODULE SEPARATOR ---
use std::cmp;

use workdir::Workdir;

use {Csv, CsvData, qcheck};

fn prop_sort(name: &str, rows: CsvData, headers: bool) -> bool {
    let wrk = Workdir::new(name);
    wrk.create("in.csv", rows.clone());

    let mut cmd = wrk.command("sort");
    cmd.arg("in.csv");
    if !headers { cmd.arg("--no-headers"); }

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let mut expected = rows.to_vecs();
    let headers = if headers && !expected.is_empty() {
        expected.remove(0)
    } else {
        vec![]
    };
    expected.sort_by(|r1, r2| iter_cmp(r1.iter(), r2.iter()));
    if !headers.is_empty() { expected.insert(0, headers); }
    rassert_eq!(got, expected)
}

#[test]
fn prop_sort_headers() {
    fn p(rows: CsvData) -> bool {
        prop_sort("prop_sort_headers", rows, true)
    }
    qcheck(p as fn(CsvData) -> bool);
}

#[test]
fn prop_sort_no_headers() {
    fn p(rows: CsvData) -> bool {
        prop_sort("prop_sort_no_headers", rows, false)
    }
    qcheck(p as fn(CsvData) -> bool);
}

#[test]
fn sort_select() {
    let wrk = Workdir::new("sort_select");
    wrk.create("in.csv", vec![svec!["1", "b"], svec!["2", "a"]]);

    let mut cmd = wrk.command("sort");
    cmd.arg("--no-headers").args(&["--select", "2"]).arg("in.csv");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![svec!["2", "a"], svec!["1", "b"]];
    assert_eq!(got, expected);
}

#[test]
fn sort_numeric() {
    let wrk = Workdir::new("sort_numeric");
    wrk.create("in.csv", vec![
        svec!["N", "S"],
        svec!["10", "a"],
        svec!["LETTER", "b"],
        svec!["2", "c"],
        svec!["1", "d"],
    ]);

    let mut cmd = wrk.command("sort");
    cmd.arg("-N").arg("in.csv");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["N", "S"],
        //Non-numerics should be put first
        svec!["LETTER", "b"],
        svec!["1", "d"],
        svec!["2", "c"],
        svec!["10", "a"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn sort_numeric_non_natural() {
    let wrk = Workdir::new("sort_numeric_non_natural");
    wrk.create("in.csv", vec![
        svec!["N", "S"],
        svec!["8.33", "a"],
        svec!["5", "b"],
        svec!["LETTER", "c"],
        svec!["7.4", "d"],
        svec!["3.33", "e"],
    ]);

    let mut cmd = wrk.command("sort");
    cmd.arg("-N").arg("in.csv");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["N", "S"],
        //Non-numerics should be put first
        svec!["LETTER", "c"],
        svec!["3.33", "e"],
        svec!["5", "b"],
        svec!["7.4", "d"],
        svec!["8.33", "a"],
    ];
    assert_eq!(got, expected);
}

#[test]
fn sort_reverse() {
    let wrk = Workdir::new("sort_reverse");
    wrk.create("in.csv", vec![
        svec!["R", "S"],
        svec!["1", "b"],
        svec!["2", "a"],
    ]);

    let mut cmd = wrk.command("sort");
    cmd.arg("-R").arg("--no-headers").arg("in.csv");

    let got: Vec<Vec<String>> = wrk.read_stdout(&mut cmd);
    let expected = vec![
        svec!["R", "S"],
        svec!["2", "a"],
        svec!["1", "b"],
    ];
    assert_eq!(got, expected);
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

--- MODULE SEPARATOR ---
use std::borrow::ToOwned;

use workdir::Workdir;

macro_rules! split_eq {
    ($wrk:expr, $path:expr, $expected:expr) => (
        // assert_eq!($wrk.path($path).into_os_string().into_string().unwrap(),
                   // $expected.to_owned());
        assert_eq!($wrk.from_str::<String>(&$wrk.path($path)),
                   $expected.to_owned());
    );
}

fn data(headers: bool) -> Vec<Vec<String>> {
    let mut rows = vec![
        svec!["a", "b"], svec!["c", "d"],
        svec!["e", "f"], svec!["g", "h"],
        svec!["i", "j"], svec!["k", "l"],
    ];
    if headers { rows.insert(0, svec!["h1", "h2"]); }
    rows
}

#[test]
fn split_zero() {
    let wrk = Workdir::new("split_zero");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "0"]).arg(&wrk.path(".")).arg("in.csv");
    wrk.assert_err(&mut cmd);
}

#[test]
fn split() {
    let wrk = Workdir::new("split");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "2"]).arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
h1,h2
a,b
c,d
");
    split_eq!(wrk, "2.csv", "\
h1,h2
e,f
g,h
");
    split_eq!(wrk, "4.csv", "\
h1,h2
i,j
k,l
");
    assert!(!wrk.path("6.csv").exists());
}

#[test]
fn split_idx() {
    let wrk = Workdir::new("split_idx");
    wrk.create_indexed("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "2"]).arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
h1,h2
a,b
c,d
");
    split_eq!(wrk, "2.csv", "\
h1,h2
e,f
g,h
");
    split_eq!(wrk, "4.csv", "\
h1,h2
i,j
k,l
");
    assert!(!wrk.path("6.csv").exists());
}

#[test]
fn split_no_headers() {
    let wrk = Workdir::new("split_no_headers");
    wrk.create("in.csv", data(false));

    let mut cmd = wrk.command("split");
    cmd.args(&["--no-headers", "--size", "2"])
       .arg(&wrk.path("."))
       .arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
a,b
c,d
");
    split_eq!(wrk, "2.csv", "\
e,f
g,h
");
    split_eq!(wrk, "4.csv", "\
i,j
k,l
");
}

#[test]
fn split_no_headers_idx() {
    let wrk = Workdir::new("split_no_headers_idx");
    wrk.create_indexed("in.csv", data(false));

    let mut cmd = wrk.command("split");
    cmd.args(&["--no-headers", "--size", "2"])
       .arg(&wrk.path("."))
       .arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
a,b
c,d
");
    split_eq!(wrk, "2.csv", "\
e,f
g,h
");
    split_eq!(wrk, "4.csv", "\
i,j
k,l
");
}

#[test]
fn split_one() {
    let wrk = Workdir::new("split_one");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "1"]).arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
h1,h2
a,b
");
    split_eq!(wrk, "1.csv", "\
h1,h2
c,d
");
    split_eq!(wrk, "2.csv", "\
h1,h2
e,f
");
    split_eq!(wrk, "3.csv", "\
h1,h2
g,h
");
    split_eq!(wrk, "4.csv", "\
h1,h2
i,j
");
    split_eq!(wrk, "5.csv", "\
h1,h2
k,l
");
}

#[test]
fn split_one_idx() {
    let wrk = Workdir::new("split_one_idx");
    wrk.create_indexed("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "1"]).arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
h1,h2
a,b
");
    split_eq!(wrk, "1.csv", "\
h1,h2
c,d
");
    split_eq!(wrk, "2.csv", "\
h1,h2
e,f
");
    split_eq!(wrk, "3.csv", "\
h1,h2
g,h
");
    split_eq!(wrk, "4.csv", "\
h1,h2
i,j
");
    split_eq!(wrk, "5.csv", "\
h1,h2
k,l
");
}

#[test]
fn split_uneven() {
    let wrk = Workdir::new("split_uneven");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "4"]).arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
h1,h2
a,b
c,d
e,f
g,h
");
    split_eq!(wrk, "4.csv", "\
h1,h2
i,j
k,l
");
}

#[test]
fn split_uneven_idx() {
    let wrk = Workdir::new("split_uneven_idx");
    wrk.create_indexed("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "4"]).arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    split_eq!(wrk, "0.csv", "\
h1,h2
a,b
c,d
e,f
g,h
");
    split_eq!(wrk, "4.csv", "\
h1,h2
i,j
k,l
");
}

#[test]
fn split_custom_filename() {
    let wrk = Workdir::new("split");
    wrk.create("in.csv", data(true));

    let mut cmd = wrk.command("split");
    cmd.args(&["--size", "2"])
       .args(&["--filename", "prefix-{}.csv"])
       .arg(&wrk.path(".")).arg("in.csv");
    wrk.run(&mut cmd);

    assert!(wrk.path("prefix-0.csv").exists());
    assert!(wrk.path("prefix-2.csv").exists());
    assert!(wrk.path("prefix-4.csv").exists());
}

--- MODULE SEPARATOR ---
use std::borrow::ToOwned;
use std::cmp;
use std::process;

use workdir::Workdir;

macro_rules! stats_tests {
    ($name:ident, $field:expr, $rows:expr, $expect:expr) => (
        stats_tests!($name, $field, $rows, $expect, false);
    );
    ($name:ident, $field:expr, $rows:expr, $expect:expr, $nulls:expr) => (
        mod $name {
            use super::test_stats;

            stats_test_headers!($name, $field, $rows, $expect, $nulls);
            stats_test_no_headers!($name, $field, $rows, $expect, $nulls);
        }
    );
}

macro_rules! stats_test_headers {
    ($name:ident, $field:expr, $rows:expr, $expect:expr) => (
        stats_test_headers!($name, $field, $rows, $expect, false);
    );
    ($name:ident, $field:expr, $rows:expr, $expect:expr, $nulls:expr) => (
        #[test]
        fn headers_no_index() {
            let name = concat!(stringify!($name), "_headers_no_index");
            test_stats(name, $field, $rows, $expect, true, false, $nulls);
        }

        #[test]
        fn headers_index() {
            let name = concat!(stringify!($name), "_headers_index");
            test_stats(name, $field, $rows, $expect, true, true, $nulls);
        }
    );
}

macro_rules! stats_test_no_headers {
    ($name:ident, $field:expr, $rows:expr, $expect:expr) => (
        stats_test_no_headers!($name, $field, $rows, $expect, false);
    );
    ($name:ident, $field:expr, $rows:expr, $expect:expr, $nulls:expr) => (
        #[test]
        fn no_headers_no_index() {
            let name = concat!(stringify!($name), "_no_headers_no_index");
            test_stats(name, $field, $rows, $expect, false, false, $nulls);
        }

        #[test]
        fn no_headers_index() {
            let name = concat!(stringify!($name), "_no_headers_index");
            test_stats(name, $field, $rows, $expect, false, true, $nulls);
        }
    );
}

fn test_stats<S>(name: S, field: &str, rows: &[&str], expected: &str,
                 headers: bool, use_index: bool, nulls: bool)
        where S: ::std::ops::Deref<Target=str> {
    let (wrk, mut cmd) = setup(name, rows, headers, use_index, nulls);
    let field_val = get_field_value(&wrk, &mut cmd, field);
    // Only compare the first few bytes since floating point arithmetic
    // can mess with exact comparisons.
    let len = cmp::min(10, cmp::min(field_val.len(), expected.len()));
    assert_eq!(&field_val[0..len], &expected[0..len]);
}

fn setup<S>(name: S, rows: &[&str], headers: bool,
            use_index: bool, nulls: bool) -> (Workdir, process::Command)
        where S: ::std::ops::Deref<Target=str> {
    let wrk = Workdir::new(&name);
    let mut data: Vec<Vec<String>> =
        rows.iter().map(|&s| vec![s.to_owned()]).collect();
    if headers { data.insert(0, svec!["header"]); }
    if use_index {
        wrk.create_indexed("in.csv", data);
    } else {
        wrk.create("in.csv", data);
    }

    let mut cmd = wrk.command("stats");
    cmd.arg("in.csv");
    if !headers { cmd.arg("--no-headers"); }
    if nulls { cmd.arg("--nulls"); }

    (wrk, cmd)
}

fn get_field_value(wrk: &Workdir, cmd: &mut process::Command, field: &str)
                  -> String {
    if field == "median" { cmd.arg("--median"); }
    if field == "cardinality" { cmd.arg("--cardinality"); }
    if field == "mode" { cmd.arg("--mode"); }

    let mut rows: Vec<Vec<String>> = wrk.read_stdout(cmd);
    let headers = rows.remove(0);
    for row in rows.iter() {
        for (h, val) in headers.iter().zip(row.iter()) {
            if &**h == field {
                return val.clone();
            }
        }
    }
    panic!("BUG: Could not find field '{}' in headers '{:?}' \
            for command '{:?}'.", field, headers, cmd);
}

stats_tests!(stats_infer_unicode, "type", &["a"], "Unicode");
stats_tests!(stats_infer_int, "type", &["1"], "Integer");
stats_tests!(stats_infer_float, "type", &["1.2"], "Float");
stats_tests!(stats_infer_null, "type", &[""], "NULL");
stats_tests!(stats_infer_unicode_null, "type", &["a", ""], "Unicode");
stats_tests!(stats_infer_int_null, "type", &["1", ""], "Integer");
stats_tests!(stats_infer_float_null, "type", &["1.2", ""], "Float");
stats_tests!(stats_infer_null_unicode, "type", &["", "a"], "Unicode");
stats_tests!(stats_infer_null_int, "type", &["", "1"], "Integer");
stats_tests!(stats_infer_null_float, "type", &["", "1.2"], "Float");
stats_tests!(stats_infer_int_unicode, "type", &["1", "a"], "Unicode");
stats_tests!(stats_infer_unicode_int, "type", &["a", "1"], "Unicode");
stats_tests!(stats_infer_int_float, "type", &["1", "1.2"], "Float");
stats_tests!(stats_infer_float_int, "type", &["1.2", "1"], "Float");
stats_tests!(stats_infer_null_int_float_unicode, "type",
             &["", "1", "1.2", "a"], "Unicode");

stats_tests!(stats_no_mean, "mean", &["a"], "");
stats_tests!(stats_no_stddev, "stddev", &["a"], "");
stats_tests!(stats_no_median, "median", &["a"], "");
stats_tests!(stats_no_mode, "mode", &["a", "b"], "N/A");

stats_tests!(stats_null_mean, "mean", &[""], "");
stats_tests!(stats_null_stddev, "stddev", &[""], "");
stats_tests!(stats_null_median, "median", &[""], "");
stats_tests!(stats_null_mode, "mode", &[""], "N/A");

stats_tests!(stats_includenulls_null_mean, "mean", &[""], "", true);
stats_tests!(stats_includenulls_null_stddev, "stddev", &[""], "", true);
stats_tests!(stats_includenulls_null_median, "median", &[""], "", true);
stats_tests!(stats_includenulls_null_mode, "mode", &[""], "N/A", true);

stats_tests!(stats_includenulls_mean,
             "mean", &["5", "", "15", "10"], "7.5", true);

stats_tests!(stats_sum_integers, "sum", &["1", "2"], "3");
stats_tests!(stats_sum_floats, "sum", &["1.5", "2.8"], "4.3");
stats_tests!(stats_sum_mixed1, "sum", &["1.5", "2"], "3.5");
stats_tests!(stats_sum_mixed2, "sum", &["2", "1.5"], "3.5");
stats_tests!(stats_sum_mixed3, "sum", &["1.5", "hi", "2.8"], "4.3");
stats_tests!(stats_sum_nulls1, "sum", &["1", "", "2"], "3");
stats_tests!(stats_sum_nulls2, "sum", &["", "1", "2"], "3");

stats_tests!(stats_min, "min", &["2", "1.1"], "1.1");
stats_tests!(stats_max, "max", &["2", "1.1"], "2");
stats_tests!(stats_min_mix, "min", &["2", "a", "1.1"], "1.1");
stats_tests!(stats_max_mix, "max", &["2", "a", "1.1"], "a");
stats_tests!(stats_min_null, "min", &["", "2", "1.1"], "1.1");
stats_tests!(stats_max_null, "max", &["2", "1.1", ""], "2");

stats_tests!(stats_len_min, "min_length", &["aa", "a"], "1");
stats_tests!(stats_len_max, "max_length", &["a", "aa"], "2");
stats_tests!(stats_len_min_null, "min_length", &["", "aa", "a"], "0");
stats_tests!(stats_len_max_null, "max_length", &["a", "aa", ""], "2");

stats_tests!(stats_mean, "mean", &["5", "15", "10"], "10");
stats_tests!(stats_stddev, "stddev", &["1", "2", "3"], "0.816496580927726");
stats_tests!(stats_mean_null, "mean", &["", "5", "15", "10"], "10");
stats_tests!(stats_stddev_null, "stddev", &["1", "2", "3", ""],
             "0.816496580927726");
stats_tests!(stats_mean_mix, "mean", &["5", "15.1", "9.9"], "10");
stats_tests!(stats_stddev_mix, "stddev", &["1", "2.1", "2.9"],
             "0.7788880963698614");

stats_tests!(stats_cardinality, "cardinality", &["a", "b", "a"], "2");
stats_tests!(stats_mode, "mode", &["a", "b", "a"], "a");
stats_tests!(stats_mode_null, "mode", &["", "a", "b", "a"], "a");
stats_tests!(stats_median, "median", &["1", "2", "3"], "2");
stats_tests!(stats_median_null, "median", &["", "1", "2", "3"], "2");
stats_tests!(stats_median_even, "median", &["1", "2", "3", "4"], "2.5");
stats_tests!(stats_median_even_null, "median",
             &["", "1", "2", "3", "4"], "2.5");
stats_tests!(stats_median_mix, "median", &["1", "2.5", "3"], "2.5");

mod stats_infer_nothing {
    // Only test CSV data with headers.
    // Empty CSV data with no headers won't produce any statistical analysis.
    use super::test_stats;
    stats_test_headers!(stats_infer_nothing, "type", &[], "NULL");
}

mod stats_zero_cardinality {
    use super::test_stats;
    stats_test_headers!(stats_zero_cardinality, "cardinality", &[], "0");
}

mod stats_zero_mode {
    use super::test_stats;
    stats_test_headers!(stats_zero_mode, "mode", &[], "N/A");
}

mod stats_zero_mean {
    use super::test_stats;
    stats_test_headers!(stats_zero_mean, "mean", &[], "");
}

mod stats_zero_median {
    use super::test_stats;
    stats_test_headers!(stats_zero_median, "median", &[], "");
}

mod stats_header_fields {
    use super::test_stats;
    stats_test_headers!(stats_header_field_name, "field", &["a"], "header");
    stats_test_no_headers!(stats_header_no_field_name, "field", &["a"], "0");
}

--- MODULE SEPARATOR ---
use workdir::Workdir;

fn data() -> Vec<Vec<String>> {
    vec![
        svec!["h1", "h2", "h3"],
        svec!["abcdefg", "a", "a"],
        svec!["a", "abc", "z"],
    ]
}

#[test]
fn table() {
    let wrk = Workdir::new("table");
    wrk.create("in.csv", data());

    let mut cmd = wrk.command("table");
    cmd.arg("in.csv");

    let got: String = wrk.stdout(&mut cmd);
    assert_eq!(&*got, "\
h1       h2   h3
abcdefg  a    a
a        abc  z\
")
}

--- MODULE SEPARATOR ---
#![allow(dead_code)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;

extern crate csv;
extern crate filetime;
extern crate quickcheck;
extern crate rand;
extern crate stats;

use std::fmt;
use std::mem::transmute;
use std::ops;

use quickcheck::{Arbitrary, Gen, QuickCheck, StdGen, Testable};
use rand::{Rng, thread_rng};

macro_rules! svec[
    ($($x:expr),*) => (
        vec![$($x),*].into_iter()
                     .map(|s: &'static str| s.to_string())
                     .collect::<Vec<String>>()
    );
    ($($x:expr,)*) => (svec![$($x),*]);
];

macro_rules! rassert_eq {
    ($given:expr, $expected:expr) => ({assert_eq!($given, $expected); true});
}

mod workdir;

mod test_cat;
mod test_count;
mod test_fixlengths;
mod test_flatten;
mod test_fmt;
mod test_frequency;
mod test_headers;
mod test_index;
mod test_join;
mod test_partition;
mod test_reverse;
mod test_search;
mod test_select;
mod test_slice;
mod test_sort;
mod test_split;
mod test_stats;
mod test_table;

fn qcheck<T: Testable>(p: T) {
    QuickCheck::new().gen(StdGen::new(thread_rng(), 5)).quickcheck(p);
}

fn qcheck_sized<T: Testable>(p: T, size: usize) {
    QuickCheck::new().gen(StdGen::new(thread_rng(), size)).quickcheck(p);
}

pub type CsvVecs = Vec<Vec<String>>;

pub trait Csv {
    fn to_vecs(self) -> CsvVecs;
    fn from_vecs(CsvVecs) -> Self;
}

impl Csv for CsvVecs {
    fn to_vecs(self) -> CsvVecs { self }
    fn from_vecs(vecs: CsvVecs) -> CsvVecs { vecs }
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
struct CsvRecord(Vec<String>);

impl CsvRecord {
    fn unwrap(self) -> Vec<String> {
        let CsvRecord(v) = self;
        v
    }
}

impl ops::Deref for CsvRecord {
    type Target = [String];
    fn deref<'a>(&'a self) -> &'a [String] { &*self.0 }
}

impl fmt::Debug for CsvRecord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bytes: Vec<_> = self.iter()
                                .map(|s| s.as_bytes())
                                .collect();
        write!(f, "{:?}", bytes)
    }
}

impl Arbitrary for CsvRecord {
    fn arbitrary<G: Gen>(g: &mut G) -> CsvRecord {
        let size = { let s = g.size(); g.gen_range(1, s) };
        CsvRecord((0..size).map(|_| Arbitrary::arbitrary(g)).collect())
    }

    fn shrink(&self) -> Box<Iterator<Item=CsvRecord>+'static> {
        Box::new(self.clone().unwrap()
                     .shrink().filter(|r| r.len() > 0).map(CsvRecord))
    }
}

impl Csv for Vec<CsvRecord> {
    fn to_vecs(self) -> CsvVecs {
        unsafe { transmute(self) }
    }
    fn from_vecs(vecs: CsvVecs) -> Vec<CsvRecord> {
        unsafe { transmute(vecs) }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialOrd)]
struct CsvData {
    data: Vec<CsvRecord>,
}

impl CsvData {
    fn unwrap(self) -> Vec<CsvRecord> { self.data }

    fn len(&self) -> usize { (&**self).len() }

    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl ops::Deref for CsvData {
    type Target = [CsvRecord];
    fn deref<'a>(&'a self) -> &'a [CsvRecord] { &*self.data }
}

impl Arbitrary for CsvData {
    fn arbitrary<G: Gen>(g: &mut G) -> CsvData {
        let record_len = { let s = g.size(); g.gen_range(1, s) };
        let num_records: usize = g.gen_range(0, 100);
        CsvData{
            data: (0..num_records).map(|_| {
                CsvRecord((0..record_len)
                          .map(|_| Arbitrary::arbitrary(g))
                          .collect())
            }).collect(),
        }
    }

    fn shrink(&self) -> Box<Iterator<Item=CsvData>+'static> {
        let len = if self.is_empty() { 0 } else { self[0].len() };
        let mut rows: Vec<CsvData> =
            self.clone()
                .unwrap()
                .shrink()
                .filter(|rows| rows.iter().all(|r| r.len() == len))
                .map(|rows| CsvData { data: rows })
                .collect();
        // We should also introduce CSV data with fewer columns...
        if len > 1 {
            rows.extend(
                self.clone()
                    .unwrap()
                    .shrink()
                    .filter(|rows|
                        rows.iter().all(|r| r.len() == len - 1))
                    .map(|rows| CsvData { data: rows }));
        }
        Box::new(rows.into_iter())
    }
}

impl Csv for CsvData {
    fn to_vecs(self) -> CsvVecs { unsafe { transmute(self.data) } }
    fn from_vecs(vecs: CsvVecs) -> CsvData {
        CsvData {
            data: unsafe { transmute(vecs) },
        }
    }
}

impl PartialEq for CsvData {
    fn eq(&self, other: &CsvData) -> bool {
        (self.data.is_empty() && other.data.is_empty())
        || self.data == other.data
    }
}

--- MODULE SEPARATOR ---
use std::env;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process;
use std::str::FromStr;
use std::sync::atomic;
use std::time::Duration;

use csv;

use Csv;

static XSV_INTEGRATION_TEST_DIR: &'static str = "xit";

static NEXT_ID: atomic::AtomicUsize = atomic::ATOMIC_USIZE_INIT;

pub struct Workdir {
    root: PathBuf,
    dir: PathBuf,
    flexible: bool,
}

impl Workdir {
    pub fn new(name: &str) -> Workdir {
        let id = NEXT_ID.fetch_add(1, atomic::Ordering::SeqCst);
        let mut root = env::current_exe().unwrap()
                           .parent()
                           .expect("executable's directory")
                           .to_path_buf();
        if root.ends_with("deps") {
            root.pop();
        }
        let dir = root.join(XSV_INTEGRATION_TEST_DIR)
                      .join(name)
                      .join(&format!("test-{}", id));
        // println!("{:?}", dir);
        if let Err(err) = create_dir_all(&dir) {
            panic!("Could not create '{:?}': {}", dir, err);
        }
        Workdir { root: root, dir: dir, flexible: false }
    }

    pub fn flexible(mut self, yes: bool) -> Workdir {
        self.flexible = yes;
        self
    }

    pub fn create<T: Csv>(&self, name: &str, rows: T) {
        let mut wtr = csv::WriterBuilder::new()
            .flexible(self.flexible)
            .from_path(&self.path(name))
            .unwrap();
        for row in rows.to_vecs().into_iter() {
            wtr.write_record(row).unwrap();
        }
        wtr.flush().unwrap();
    }

    pub fn create_indexed<T: Csv>(&self, name: &str, rows: T) {
        self.create(name, rows);

        let mut cmd = self.command("index");
        cmd.arg(name);
        self.run(&mut cmd);
    }

    pub fn read_stdout<T: Csv>(&self, cmd: &mut process::Command) -> T {
        let stdout: String = self.stdout(cmd);
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(io::Cursor::new(stdout));

        let records: Vec<Vec<String>> = rdr
            .records()
            .collect::<Result<Vec<csv::StringRecord>, _>>()
            .unwrap()
            .into_iter()
            .map(|r| r.iter().map(|f| f.to_string()).collect())
            .collect();
        Csv::from_vecs(records)
    }

    pub fn command(&self, sub_command: &str) -> process::Command {
        let mut cmd = process::Command::new(&self.xsv_bin());
        cmd.current_dir(&self.dir).arg(sub_command);
        cmd
    }

    pub fn output(&self, cmd: &mut process::Command) -> process::Output {
        debug!("[{}]: {:?}", self.dir.display(), cmd);
        println!("[{}]: {:?}", self.dir.display(), cmd);
        let o = cmd.output().unwrap();
        if !o.status.success() {
            panic!("\n\n===== {:?} =====\n\
                    command failed but expected success!\
                    \n\ncwd: {}\
                    \n\nstatus: {}\
                    \n\nstdout: {}\n\nstderr: {}\
                    \n\n=====\n",
                   cmd, self.dir.display(), o.status,
                   String::from_utf8_lossy(&o.stdout),
                   String::from_utf8_lossy(&o.stderr))
        }
        o
    }

    pub fn run(&self, cmd: &mut process::Command) {
        self.output(cmd);
    }

    pub fn stdout<T: FromStr>(&self, cmd: &mut process::Command) -> T {
        let o = self.output(cmd);
        let stdout = String::from_utf8_lossy(&o.stdout);
        stdout.trim_matches(&['\r', '\n'][..]).parse().ok().expect(
            &format!("Could not convert from string: '{}'", stdout))
    }

    pub fn assert_err(&self, cmd: &mut process::Command) {
        let o = cmd.output().unwrap();
        if o.status.success() {
            panic!("\n\n===== {:?} =====\n\
                    command succeeded but expected failure!\
                    \n\ncwd: {}\
                    \n\nstatus: {}\
                    \n\nstdout: {}\n\nstderr: {}\
                    \n\n=====\n",
                   cmd, self.dir.display(), o.status,
                   String::from_utf8_lossy(&o.stdout),
                   String::from_utf8_lossy(&o.stderr));
        }
    }

    pub fn from_str<T: FromStr>(&self, name: &Path) -> T {
        let mut o = String::new();
        fs::File::open(name).unwrap().read_to_string(&mut o).unwrap();
        o.parse().ok().expect("fromstr")
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub fn xsv_bin(&self) -> PathBuf {
        self.root.join("xsv")
    }
}

impl fmt::Debug for Workdir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "path={}", self.dir.display())
    }
}

// For whatever reason, `fs::create_dir_all` fails intermittently on Travis
// with a weird "file exists" error. Despite my best efforts to get to the
// bottom of it, I've decided a try-wait-and-retry hack is good enough.
fn create_dir_all<P: AsRef<Path>>(p: P) -> io::Result<()> {
    let mut last_err = None;
    for _ in 0..10 {
        if let Err(err) = fs::create_dir_all(&p) {
            last_err = Some(err);
            ::std::thread::sleep(Duration::from_millis(500));
        } else {
            return Ok(())
        }
    }
    Err(last_err.unwrap())
}

```

**Filepath**: `./xsv/src/main.rs`

---

*Generated by code-ingest export at 2025-09-28 07:48:50 UTC*
