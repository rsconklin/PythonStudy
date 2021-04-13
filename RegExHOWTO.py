### Notes on the Python Doc: Regular Expression HOWTO

## Introduction:
# Available through the 're' module. Specify the rules for the
# set of possible strings to match. This could contain English
# sentences, email addresses, TeX commands, or anything else.
# Can then ask questions such as, "Does this string match the
# pattern?" or, "Is there a match for this pattern anywhere in
# this string? REs can also be used to modify a string or to
# split it apart in various ways.

# Regular expression patterns are compiled into a series of
# bytecodes which are then executed by a matching engine written
# in C. Advanced use for optimizing the bytecodes requires a
# good understanding of the matching engine's internals.

# The RegEx language is relatively small and restricted and
# cannot do all string processing. It may be advantageous
# (although slower) to use Python code for readability.

## Simple Patterns:
# Most common task: Matching characters.

## Matching Characters:
# Most letters and characters will match themselves.
# For example the regular
# expression 'test' will match the string 'test' exactly. There
# also a case-insensitive mode to match 'Test' or 'TEST' as well.
# Exceptions to this rule are special metacharacters that don't
# match themselves.

# The complete list of metacharacters:
# . ^ $ * + ? { } [ ] \ | ( )

# [ and ] specify a character class which is a set of characters
# to match. Characters can be listed individually, or a range can
# be indicated by giving two characters and separating them by '-'.

# Metacharacters are not active inside classes, so [akm$] will
# match any of the characters, including '$'.

# Characters not listed within the class can be matched by
# complementing the set by inserting '^' as the first character of
# the class. For example, [^5] will match any character except
# '5'. If '^' appears anywhere else it has no special meaning.

# The most important metacharacter may be the backslash \. As in
# Python string literals, \ can be followed by various characters
# to signal special sequences. It is also used to escape all the
# metacharacters so they can be matched in patterns.

# The following sequences math any...
# \d : decimal digit; equivalent to the class [0-9].
# \D : non-digit character; equivalent to [^0-9].
# \s : whitespace character; equivalent to [ \t\n\r\f\v].
# \S : non-whitespace character; equivalent to [^ \t\n\r\f\v].
# \w : alphanumeric character; equivalent to [a-zA-Z0-9_].
# \W : non-alphanumeric character; equivalent to [^a-zA-Z0-9_].

# These can be included inside character classes. For example:
# [\s,.] matches all whitespace, ',', and '.' characters.

# '.' matches anything except a newline character, and there is an
# alternate mode (re.DOTALL) that even matches the newline.

## Repeating Things:
# It is possible to specify the number of times the RE must be
# repeated.

# * specifies that the previous character can be matched zero or more
# times, instead of exactly once. For example, 'ca*t' will match 'cat'
# 'caaat' and 'caaaaaat' and 'ct'. Repetition such as * are greedy.
# They seek as far as possible for a match and back up and test fewer
# repetitions if the attempt is unsuccessful.

# + matches one or more times. 'ca+t' will match 'cat' or 'caaat' but
# not 'ct'.

# There are two more repeating qualifiers.

# '?' matches either once or zero times. It marks something as
# optional. For example, 'home0?brew' matches either 'homebrew' or
# 'home-brew'.

# The most complicated repeated qualifier is '{m, n}' where 'm' and 'n'
# are decimal integers. It means there must be m <= repetition <= n.
# For example, 'a/{1,3}b' will match 'a/b' and 'a//b' and 'a///b' but
# not 'a////b'. m and n can be omitted, in which case a reasonable
# value is assumed. Removing m is interpreted as a lower limit of 0
# while removing n is an upper bound of infinity. The three other
# qualifiers can be expressed using this notation. For example,
# * == {0,}, + == {1,} and ? == {0, 1}. It is, however, shorter and
# easier to read the simpler qualifiers.

## Using Regular Expressions:
# The 're' module provides an interface to the regular expression
# engine, allowing the compilation of REs into objects and their
# matching.

## Compiling Regular Expressions:
# Regular expressions are compiled into pattern objects which have
# methods for various operations such as searching for pattern matches
# and performing string substitutions.

import re
p = re.compile('ab*')
print(p)

p = re.compile('ab*', re.IGNORECASE)

# Putting REs in strings keeps Python simpler (REs are not part of
# the core language), but there is one disadvantage...

## The Backslash Plague:
# Backslashes are used in REs to remove the special effect of
# metacharacters, but this conflicts with Python's usage for them
# resulting in potentially excessively long sequences of \. The
# solution is to use Python's raw string notation for regular
# expressions. Backslashes are not handled in any special way in
# a string literal prefixed with 'r', for example in r'\n' is a
# two-character string containing \ and n, while '\n' is a
# one-character string containing a newline.

## Performing Matches:
# Once you have an object representing a compiled regular
# expression, what do you do with it? Pattern objects have several
# methods and attributes. Only the most significant ones are
# covered here, with a complete listing in the 're' docs.

# match() : Determine if the RE matches at the beginning of
#           the string.
# search() : Scan through a string, looking for any location
#            where this RE matches.
# findall() : Find all substrings where the RE matches, and
#             return them as a list.
# finditer() : Find all substrings where the RE matches, and
#              return them as an iterator.

# match() and search() return None if no match can be found. If
# they are successful, a match object instance is returned. This
# contains information about the match: where it starts and ends,
# the substring it matched, and more.

p = re.compile('[a-z]+')
m = p.match('tempo')

print(m)

# Now it is possible to query the match object instances for
# more information about the matching string. Match objects
# also have several methods and attributes. The most important
# ones are:

# group() : Return the substring matched by the RE.
# start() : Return the starting position of the match.
# end() : Return the ending position of the match.
# span() : Return a tuple containing the (start, end) positions
#          of the match.

print(m.group(), (m.start(), m.end()), m.span())

print(p.match('::: message'))
m = p.search('::: message'); print(m)
print(m.group())
print(m.span())

# The most common style is to store the match object in a
# variable, then check if it was None. p = re.compile(...),
# m = p.match('string'), if match: print, else print no match.

p = re.compile(r'\d+')
print(p.findall('1 chicken, 13 eggs, and 7 pieces of bacon.'))

iterator = p.finditer('1 chicken, 13 eggs, and 7 pieces of bacon.')
print(iterator)
for match in iterator:
    print(match.span())

## Module-Level Functions.
# So there are pattern and match objects.
# The methods match(), search(), findall(), sub(), etc., can
# be called at the module level with the regular expression
# as an argument.
print(re.match(r'From\s*', 'Fromage amk'))

# If inside a loop, precompiling the REs saves a few function
# calls.

## Compilation Flags:
# These modify how regular expressions work. They are available
# in the re module under two names: A long name such as
# IGNORECASE and a short on-letter name such as I. Multiple
# flags can be specified by bitwise OR: re.I | re.M sets both
# the I and M flags.

# Here are the available flags:
# ASCII, A : Makes several escapes like \w, \b, \s, and \d
# match only on ASCII characters with the respective property.
# DOTALL, S : Make . match any character, including newlines.
# IGNORECASE, I : Do case-insensitive matches.
# LOCALE, L : Do a locale-aware match.
# MULTILINE, M : Multi-line matching, affecting ^ and $.
# VERBOSE, X (for 'extended') : Enable verbose REs, which can
# be organized more cleanly and understandably.

## More Pattern Power:
# More metacharacters and how to use groups to retrieve portions
# of text that were matched.

## More Metacharacters:
# Most of the remaining metacharacters will be covered in this
# section.

# Some of the remaining metacharacters are zero-width
# assertions. That is, they do not cause the engine to advance
# through the string. They consume no characters at all, and
# simply succeed or fail. For example, \b is an assertion that
# the current position is located at a word boundary. The
# position isn't changed by the \b at all.

# | : Alternation, or the 'or' operator. If A and B are REs,
# A|B will match any string that matches either A or B. It has
# very low precedence to make it work reasonably when
# alternating multi-character strings. Crow|Servo will match
# either Crow or Servo but not substrings of either. To match
# a literal |, use \| or enclose it inside a character class,
# as in [|].
# ^ :  Match at the beginning of lines. For example, to match
# the word From only at the beginning of a line,
print(re.search('^From', 'From Here to Eternity').group())
# $ : Matches at the end of a line, which is defined as either
# the end of the string, or any location followed by a newline
# character.
print(re.search('}$', '{block}'))
print(re.search('}$', '{block} '))
print(re.search('}$', '{block}\n'))
# \A : Matches only at the start of the string. When not in
# MULTLINE mode, \A and ^ are effectively the same.
# \Z : Matches only at the end of a string.
# \b : Word boundary. A zero-width assertion that matches only
# at the beginning or end of a word. A word is defined as a
# sequence of alphanumeric characters, so the end of a word is
# indicated by whitespace or a non-alphanumeric character. The
# following matches 'class' only when it's an isolated word.
p = re.compile(r'\bclass\b')
print(p.search('no class at all'))
print(p.search('no classifieds at all'))
# Inside a string class, where there is no use for this
# assertion, \b represents the backspace character, for
# compatibility with Python's string literals.
# \B : Another zero-width assertion, the opposite of \b,
# matching only when the current position is not a word
# boundary.

## Grouping:
# Regular expressions are often used to dissect strings by
# writing an RE divided into several subgroups which match
# different components of interest. For example, an
# RFC-822 header line is divided into a header name and a
# value, separated by a ':', like this:
# From: author@example.com
# User-Agent: Thunderbird 1.5.0.9 (X11/20061227)
# MIME-Version: 1.0
# To: editor@example.com

# Groups are marked by the ( ) metacharacters. They have
# similar meaning as in math where they group expressions.
# The contents of a group can be repeated by a repeating
# qualifier, such as * + ? or {m, n}. For example, (ab)*
# will match zero or more repetitions of ab.
p = re.compile('(ab)*')
print(p.match('ababababababab').span())
# Groups indicated with ( ) also capture the starting and
# ending index of the text that they match. This can be
# retrieved by passing an argument to group(), start(),
# end(), and span(). Groups are numbered starting with 0.
# Group 0 is always present; it is the whole RE, so all
# match arguments have group 0 as their default argument.
p = re.compile('(a)b')
m = p.match('ab')
print(m.group())
print(m.group(0))
# Subgroups are numbered from left to right, from 1
# upward. Groups can be nested; the group number can be
# determined by counting the opening parenthesis
# characters from left to right.
p = re.compile('(a(b)c)d')
m = p.match('abcd')
print(m.group(0), m.group(1), m.group(2))
# group can be passed multiple group numbers at one time,
# in which case it will return a tuple containing the
# corresponding values for those groups.
print(m.group(2, 1, 0, 1, 2))
# The groups() method returns a tuple containing the
# strings for all the subgroups, from 1 upwards:
print(m.groups())
# Backreferences make it possible to specify that  the
# contents of an earlier capturing group must also be
# found ta the current location in the string. For
# example, a \1 will succeed if the exact contents of
# group 1 can be found at the current position, and
# fails otherwise. This conflicts with Python arbitrary
# characters, so for RegEx raw strings should be
# specified for backreferences. Detect duplicates:
p = re.compile(r'\b(\w+)\s+\1\b')
print(p.search('Chickens in the the the the pot.').group())
p = re.compile(r'(\b(\w+)\s+\2\b)')
print(p.search('Chickens in the the the the pot.').group(0, 1, 2))
# p = re.compile(r'(\b(\w+)\s+\2\b)\1') # Doesn't seem to return.
# print(p.search('Chickens in the the the the pot.').group())
# These are not often useful for just searching through
# strings, but they are *ver* useful when performing
# string substitutions.

## Non-capturing and Named Groups:
# Elaborate REs may use many groups, both to capture
# substrings of interest and to group and structure the RE
# itself. There are two features that help with this.

# To group part of a regular expression without capturing it,
# a non-capturing group may be used: (?:...), where the ...
# may be replaced with any other regular expression.
m = re.match('([abc])+', 'abc')
print(m.groups())
m = re.match('(?:[abc])+', 'abc')
print(m.group(), m.groups())

# A more significant feature is named groups; these can be
# referenced by name. The syntax for a named group is one
# of the Python-specific extensions: (?P<name>...). name is
# the name of the group. Named groups behave the same as
# capturing groups, and additionally associate a name with
# the group. The match objects methods that deal with
# capturing groups all accept either integers or names as
# strings. Named groups are still given numbers.
p = re.compile(r'(?P<word>\b\w+\b)')
m = p.search('(((( Lots of punctuation )))')
print(m.group('word'), m.group(1))
print(m.group(0, 1))
# Additionally, named groups can be retrieved as a
# dictionary with groupdict():
m = re.match(r'(?P<first>\w+) (?P<last>\w+)', 'Jane Doe')
print(m.groupdict())
# There are also backreferences to named groups using
# (?P=name), indicating that the contents of the group
# called name should again be matched at the current point.
# Find doubled words:
p = re.compile(r'\b(?P<word>\w+)\s+(?P=word)\b')
print(p.search('Chickens in the the pot.').group())

## Lookahead Assertions:
# Another zero-width assertion is the lookahead assertion.
# Lookahead assertions are available in positive and
# negative form:
# (?=...) : Positive lookahead assertion. Succeeds if the
# contained regular expression ... matches at the current
# location, and fails otherwise. However, once the
# contained expression has been tried, the matching engine
# doesn't advance at all.
# (?!...) : Negative lookahead assertion. The opposite of
# the positive assertion. Succeeds if ... does not match at
# the current position.

# For example, a base name and an extension, news.rc. The
# pattern to match this is .*[.].*$
# Trying to exclude a 'bat' extension gets very complicated,
# but this is simply matched by a negative lookahead
# assertion. .*[.](?!bat$|exe$)[^.]*$

## Modifying Strings:
# Regular expressions can modify strings in various ways
# using the following pattern methods:

# split() : Split the string into a list, splitting it
# wherever the RE matches.
# sub() : Find all substrings where the RE matches, and
# replace them with a different string.
# subn() : Does the same thing as sub(), but returns the
# new string and the number of replacements.

## Splitting Strings:
# Split with the delimiter as any sequence of
# non-alphanumeric characters:
p = re.compile(r'\W+')
print(p.split('I would like to purchase a roast of chicken.'))
print(p.split('I would like to purchase a roast of chicken.', 3))

# Capturing parentheses may be used to get the text
# between delimiters as well as the delimiters themselves.
p = re.compile(r'\W+')
p2 = re.compile(r'(\W+)')
print(p.split('This... is a *** test!!!!'))
print(p2.split('This... is a *** test!!!!'))
# A module-level function is also available.

## Search and Replace:
# Replace colour names with the word colour:
p = re.compile('(red|white|blue)')
print(p.sub('colour', 'Oooh, the red white blue!'))
print(p.sub('colour', 'Oooh, the red white blue!', 2))

# subn() does the same work but returns a 2-tuple
# containing the new string value and the number
# of replacements that were performed:
p = re.compile('(red|white|blue)')
print(p.subn('colour', 'Oooh, the red white blue!'))
print(p.subn('colour', 'Oooh, the red white blue!', 2))

# Empty matches example.

# \g<name> and \g<number> can be used to replace
# by the group name and number, respectively.

# Functions can also be called to replace values.
# For example,
p = re.compile(r'\d+')
print(p.sub('test', 'Square 1 these 2 numbers 3, sir.'))
print(p.sub(lambda x: str(int(x.group())**2), 'Square 1 these 2 numbers 3, sir.'))

# To use a flag in the module-level function,
# either a pattern object must be passed as the
# first parameter, or an embedded modifier must be
# used.
print(re.sub('(?i)b+', 'x', 'bbbb BBBB'))

## Common Problems:

## Use String Methods:
# Sometimes it is not necessary to use the full
# power of RegEx, with its flags such as IGNORECASE,
# and the Python string methods may be used instead.
# They are usually much faster because the
# implementation is a single small C loop that has
# been optimized for the purpose, instead of the
# large, more generalized regular expression engine.

## match() versus search():
# match() only checks if the RE matches at the
# beginning of the string while search() scans
# forward for a match.
# Using .* instead of search() kills the optimization
# during compilation, causing the search to be done

## Greedy versus Non-Greedy:
# Greedy qualifiers may match the start with the end
# despite the pattern being found earlier, since
# intermediate material may get absorbed by a
# metacharacter function such as .*. The solution may
# be to use non-greedy qualifiers *?, +?, ??, or
# {m, n}? which match as little text as possible.
# A match is sought, and if it fails, the engine
# advances a character at a time, retrying the match
# at every step.

## Using re.VERBOSE:

