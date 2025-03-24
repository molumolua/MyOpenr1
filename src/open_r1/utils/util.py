import pprint
import re
from collections import defaultdict

def last_boxed_only(sample):
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    

def find_position(section, next_section, section_list, title_list, matches, answer_len, logger, begin=0):
    try:
        if logger:
            logger.debug(f"Finding position for section '{section}' and next_section '{next_section}' starting from index {begin}")
        start = -1
        end = -1
        # 查找当前章节的位置
        for i in range(len(title_list)-1,-1,-1):
            if title_list[i] == section_list[section]:
                start = matches[i].end()
                if logger:
                    logger.debug(f"Found start of section '{section_list[section]}' at position {start}")
                break
        if start == -1:
            if logger:
                logger.error(f"Section '{section_list[section]}' not found in the title list.")
            return start, end, begin

        # 查找下一个章节的位置（如果有的话）
        if next_section < len(section_list):
            for i in range(len(title_list)-1,-1,-1):
                if title_list[i] == section_list[next_section]:
                    end = matches[i].start()
                    if logger:
                        logger.debug(f"Found end of section '{section_list[section]}' at position {end}")
                    break
        else:
            end = answer_len  # 如果没有下一个章节，则使用提供的 `answer_len` 作为结束位置。
            if logger:
                logger.debug(f"No next section. Using answer_len {answer_len} as end position.")

        # # 查找当前章节的位置
        # for i in range(begin, len(title_list)):
        #     if title_list[i] == section_list[section]:
        #         begin = i + 1
        #         start = matches[i].end()
        #         logger.debug(f"Found start of section '{section_list[section]}' at position {start}")
        #         break
        # if start == -1:
        #     logger.error(f"Section '{section_list[section]}' not found in the title list.")
        #     return start, end, begin

        # # 查找下一个章节的位置（如果有的话）
        # if next_section < len(section_list):
        #     for i in range(begin, len(title_list)):
        #         if title_list[i] == section_list[next_section]:
        #             begin = i
        #             end = matches[i].start()
        #             logger.debug(f"Found end of section '{section_list[section]}' at position {end}")
        #             break
        # else:
        #     end = answer_len  # 如果没有下一个章节，则使用提供的 `answer_len` 作为结束位置。
        #     logger.debug(f"No next section. Using answer_len {answer_len} as end position.")

        if end == -1 and next_section < len(section_list):
            if logger:
                logger.error(f"Next section '{section_list[next_section]}' not found in the title list.")
        return start, end, begin
    except Exception as e:
        if logger:
            logger.error(f"Error in find_position: {e}")
        return -1, -1, begin

def clear_string(s):
    return s[:-1] if s and s[-1] == ':' else s

def parse_answer(answer_text, sections, logger):
    try:
        if logger:
            logger.debug("Parsing answer text.")
        extracted = {section: "" for section in sections}

        # 匹配标题部分的正则表达式（支持以###或更多#开头的格式）
        pattern = re.compile(
            r'^\s*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?\*\*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?(.*?)\*\*:?', re.MULTILINE
        )
        matches = list(pattern.finditer(answer_text))

        # 如果没有找到匹配，再尝试另一种格式的正则表达式
        if not matches:
            pattern = re.compile(
                r'^\s*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?\*\*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?(.*?):\*\*', re.MULTILINE
            )
            matches = list(pattern.finditer(answer_text))

        # 如果仍没有匹配，返回空内容
        if not matches:
            logger.warning("No section headers matched in the answer text.")
            logger.warning(answer_text)
            return ("" for _ in sections)

        if logger:
            logger.debug(f"Found {len(matches)} section headers.")
        begin = 0
        title_list = [clear_string(match.group(1).strip()) for match in matches]
        for idx, section in enumerate(sections):
            start, end, begin = find_position(idx, idx + 1, sections, title_list, matches, len(answer_text), logger, begin)
            if start == -1 or end == -1:
                if logger:
                    logger.warning(f"Could not extract section '{section}'.")
                continue
            # 提取内容并去除前后空白字符
            content = answer_text[start:end].strip()
            extracted[section] = content
            if logger:
                logger.debug(f"Extracted content for section '{section}': {content[:50]}...")  # 只显示前50个字符
        return (extracted[section] for section in sections)
    except Exception as e:
        if logger:
            logger.error(f"Error in parse_answer: {e}")
        return ("" for _ in sections)

def extract_think_and_after(text):
    """
    提取字符串中 <think> 标签内部的内容，以及 </think> 之后的文本。

    参数：
        text (str): 包含 <think> 标签的完整字符串。

    返回：
        tuple: (think_content, after_think)
               think_content 为 <think>...</think> 中的文本（若没匹配到返回 None）。
               after_think 为 </think> 后的文本（若没匹配到返回 None）。
    """
    # 使用正则表达式，启用 DOTALL (re.DOTALL) 使 '.' 能匹配换行符
    if "<think>" not in text:
        text = "<think>" + text
    pattern = r"<think>(.*?)</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        after_think = match.group(2).strip()
        return think_content, after_think
    else:
        # 如果没有匹配到，就返回 (None, None)
        return None, None
    



def process_output_data(data_list):
    # 使用 defaultdict 来聚合
    grouped = defaultdict(list)

    # 遍历数据，将相同 original_problem 的 dict 聚集在一起
    for item in data_list:
        grouped[item['original_problem']].append(item)

    # 转换成二维 list
    result = list(grouped.values())
    return result