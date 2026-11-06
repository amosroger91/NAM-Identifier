# Guitar isolation utilities

import os
import subprocess
import sys
from typing import List, Optional

def demucs_command(python_exe: str, input_wav: str, out_dir: str) -> List[str]:
    """
    Build a Demucs command to separate guitar from input audio.
    
    Args:
        python_exe: Path to Python executable
        input_wav: Input WAV file path
        out_dir: Output directory
        
    Returns:
        List of command arguments
    """
    # Use htdemucs_6s model and keep guitar stem
    return [
        python_exe,
        '-m', 'demucs',
        '--model', 'htdemucs_6s',
        '--mp3',
        '--stem', 'guitar',
        '-o', out_dir,
        input_wav
    ]


def find_guitar_stem(demucs_output_dir: str) -> Optional[str]:
    """
    Find the guitar stem file from Demucs output.
    
    Args:
        demucs_output_dir: Directory containing Demucs output
        
    Returns:
        Path to guitar stem file, or None if not found
    """
    # Look for files containing 'guitar' in the name
    try:
        for root, dirs, files in os.walk(demucs_output_dir):
            for file in files:
                if 'guitar' in file.lower() and file.endswith(('.wav', '.mp3')):
                    return os.path.join(root, file)
    except Exception:
        pass
    
    return None


def isolate_guitar(input_wav: str, output_dir: str) -> Optional[str]:
    """
    Isolate guitar from input audio using Demucs.
    
    Args:
        input_wav: Input WAV file path
        output_dir: Output directory
        
    Returns:
        Path to isolated guitar file, or None if failed
    """
    try:
        # Try to import demucs
        import demucs
        
        # Build command
        cmd = demucs_command(sys.executable, input_wav, output_dir)
        
        # Run Demucs
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find guitar stem
        return find_guitar_stem(output_dir)
        
    except ImportError:
        # If demucs is not installed, return error
        return None
    except Exception as e:
        # Handle other errors
        return None