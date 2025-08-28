import React, { useState } from 'react';
import { Box, Typography, TextField, Button, CircularProgress, MenuItem, Paper } from '@mui/material';
import { useTranslation } from 'react-i18next';

const languageOptions = [
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'Hindi' },
  { value: 'te', label: 'Telugu' },
  { value: 'ta', label: 'Tamil' },
  { value: 'kn', label: 'Kannada' },
  { value: 'bn', label: 'Bengali' },
  { value: 'mr', label: 'Marathi' },
  { value: 'pa', label: 'Punjabi' },
  { value: 'gu', label: 'Gujarati' },
  { value: 'ml', label: 'Malayalam' },
  { value: 'or', label: 'Odia' },
];

const getToken = () => localStorage.getItem('token');

async function getDietRecommendation(prompt, language) {
  const token = getToken();
  const res = await fetch('/diet-recommendation/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ prompt, language }),
  });
  if (!res.ok) throw new Error('Failed to get diet recommendation');
  return await res.json();
}

const DietRecommendation = () => {
  const { t } = useTranslation();
  const [language, setLanguage] = useState('en');
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult('');
    try {
      const res = await getDietRecommendation(prompt, language);
      setResult(res.recommendation);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <Box sx={{ minHeight: '80vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', pt: 4 }}>
      <Paper sx={{ maxWidth: 500, width: '100%', mx: 'auto', p: 4, borderRadius: 3, boxShadow: 3 }}>
        <Typography variant="h5" align="center" gutterBottom>
          {t('diet_recommendation')}
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center" gutterBottom>
          {t('your_diet_question') + ' ' + t('select_language') + ' ' + t('get_recommendation').toLowerCase() + '.'}
        </Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            label={t('your_diet_question')}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            fullWidth
            required
            sx={{ mb: 2 }}
          />
          <TextField
            select
            label={t('select_language')}
            value={language}
            onChange={e => setLanguage(e.target.value)}
            sx={{ mb: 2, width: '100%' }}
          >
            {languageOptions.map(opt => (
              <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
            ))}
          </TextField>
          <Button type="submit" variant="contained" fullWidth disabled={loading || !prompt} sx={{ mb: 2 }}>
            {loading ? <CircularProgress size={24} /> : t('get_recommendation')}
          </Button>
        </form>
        {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>}
        {result && (
          <Paper sx={{ mt: 4, p: 2 }}>
            <Typography variant="subtitle1" color="primary">{t('recommendation')}</Typography>
            <Typography>{result}</Typography>
          </Paper>
        )}
      </Paper>
    </Box>
  );
};

export default DietRecommendation; 