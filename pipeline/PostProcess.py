import re
import unicodedata

class PostProcess:    
    def __init__(self):
        # Ligature mappings (common typographic ligatures)
        self.LIGATURES = {
            'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'ﬅ': 'st', 'ﬆ': 'st', 'Ĳ': 'IJ', 'ĳ': 'ij', 'Œ': 'OE', 'œ': 'oe',
            'Æ': 'AE', 'æ': 'ae'
        }
        
        # Quote normalization (fancy → standard)
        self.QUOTES = {
            '"': '"', '"': '"', '„': '"', '‟': '"',
            ''': "'", ''': "'", '‚': "'", '‛': "'",
            '«': '"', '»': '"',
        }
        
        # Dash normalization
        self.DASHES = {
            '–': '-',   # En dash
            '—': '--',  # Em dash
            '−': '-',   # Minus sign
            '‐': '-',   # Hyphen
            '‑': '-',   # Non-breaking hyphen
            '‒': '-',   # Figure dash
        }
        
        # Bullet normalization
        self.BULLETS = {
            '•': '-', '◦': '-', '▪': '-', '▫': '-',
            '‣': '-', '⁃': '-', '●': '-', '○': '-',
            '■': '-', '□': '-', '◆': '-', '◇': '-',
        }

    def process(self, text):
        # 1. Low-level Unicode cleanup
        text = unicodedata.normalize('NFKC', text)
        text = self._remove_control_chars(text)
        text = self._remove_soft_hyphens(text)
        
        # 2. Character normalization
        text = self._normalize_ligatures(text)
        text = self._normalize_quotes(text)
        text = self._normalize_dashes(text)
        text = self._normalize_bullets(text)
        
        # 3. Whitespace cleanup
        text = self._normalize_whitespace(text)
        
        # 4. Content-aware fixes
        text = self._heal_hyphenation(text)
        text = self._fix_punctuation_spacing(text)
        
        return text
    
    def _remove_control_chars(self, text):
        """Remove invisible control characters (U+0000 to U+001F, except newline/tab)."""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    def _remove_soft_hyphens(self, text):
        """Remove soft hyphens and other invisible formatting chars."""
        return re.sub(r'[\u00AD\u200B\u200C\u200D\uFEFF]', '', text)
    
    def _normalize_ligatures(self, text):
        """Expand typographic ligatures to their base characters."""
        for lig, expansion in self.LIGATURES.items():
            text = text.replace(lig, expansion)
        return text
    
    def _normalize_quotes(self, text):
        """Convert fancy quotes to standard ASCII quotes."""
        for fancy, standard in self.QUOTES.items():
            text = text.replace(fancy, standard)
        return text
    
    def _normalize_dashes(self, text):
        """Normalize various dash characters."""
        for dash, standard in self.DASHES.items():
            text = text.replace(dash, standard)
        return text
    
    def _normalize_bullets(self, text):
        """Standardize bullet point characters."""
        for bullet, standard in self.BULLETS.items():
            text = text.replace(bullet, standard)
        return text
    
    def _normalize_whitespace(self, text):
        """Normalize all whitespace characters."""
        # Unicode spaces → regular space
        text = re.sub(r'[\u00A0\u2000-\u200A\u202F\u205F\u3000]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        # Normalize line endings
        text = re.sub(r'\r\n?', '\n', text)
        # Collapse multiple newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _heal_hyphenation(self, text):
        """Rejoin words split by hyphenation at line breaks."""
        # "exam-\nple" → "example"
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # "exam- ple" → "example"
        text = re.sub(r'(\w+)-\s+([a-z])', r'\1\2', text)
        return text
    
    def _fix_punctuation_spacing(self, text):
        """Ensure proper spacing around punctuation."""
        # Add space after . ! ? if followed by uppercase (any language)
        # Use function to check Unicode category for uppercase letters
        def add_space_after_punct(match):
            punct = match.group(1)
            next_char = match.group(2)
            # Check if next char is uppercase in any language
            if unicodedata.category(next_char) == 'Lu':
                return punct + ' ' + next_char
            return match.group(0)
        
        # Match punctuation followed by any letter
        text = re.sub(r'([.!?])(\w)', add_space_after_punct, text)
        # Remove space before punctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        # Ensure space after comma/semicolon if followed by letter (any language)
        text = re.sub(r'([,;])(\w)', r'\1 \2', text)
        return text

