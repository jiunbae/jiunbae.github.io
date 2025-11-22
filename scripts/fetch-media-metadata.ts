#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import matter from 'gray-matter';
import sharp from 'sharp';

const REVIEWS_DIR = path.resolve(__dirname, '../contents/reviews');
const TMDB_API_KEY = process.env.TMDB_API_KEY || '';
const GOOGLE_BOOKS_API_KEY = process.env.GOOGLE_BOOKS_API_KEY || '';

interface ReviewFrontmatter {
  title: string;
  mediaType: 'movie' | 'series' | 'animation' | 'book';
  rating?: number;
  oneLiner?: string;
  date: string;
  slug: string;
  tags?: string[];
  poster?: string;
  metadata?: {
    originalTitle?: string;
    year?: number;
    director?: string;
    creator?: string;
    author?: string;
    genre?: string[];
    runtime?: string;
    pages?: string;
  };
  externalIds?: {
    tmdbId?: string;
    isbn?: string;
  };
}

async function fetchTMDBMetadata(tmdbId: string, mediaType: 'movie' | 'series' | 'animation') {
  try {
    const endpoint = mediaType === 'series' ? 'tv' : 'movie';
    const url = `https://api.themoviedb.org/3/${endpoint}/${tmdbId}?api_key=${TMDB_API_KEY}&language=ko-KR`;

    const response = await axios.get(url);
    const data = response.data;

    let runtime = '';
    if (mediaType === 'movie' || mediaType === 'animation') {
      runtime = data.runtime ? `${data.runtime}분` : '';
    } else {
      runtime = data.number_of_seasons ? `시즌 ${data.number_of_seasons}` : '';
      if (data.number_of_episodes) {
        runtime += `, 총 ${data.number_of_episodes}화`;
      }
    }

    return {
      originalTitle: data.original_title || data.original_name,
      year: data.release_date?.substring(0, 4) || data.first_air_date?.substring(0, 4),
      director: data.credits?.crew?.find((c: any) => c.job === 'Director')?.name,
      creator: data.created_by?.[0]?.name,
      genre: data.genres?.map((g: any) => g.name) || [],
      runtime,
      posterUrl: data.poster_path ? `https://image.tmdb.org/t/p/w500${data.poster_path}` : null,
    };
  } catch (error) {
    console.error(`Failed to fetch TMDB metadata for ID ${tmdbId}:`, error);
    return null;
  }
}

async function fetchGoogleBooksMetadata(isbn: string) {
  try {
    const url = `https://www.googleapis.com/books/v1/volumes?q=isbn:${isbn}${GOOGLE_BOOKS_API_KEY ? `&key=${GOOGLE_BOOKS_API_KEY}` : ''}`;

    const response = await axios.get(url);
    const data = response.data;

    if (!data.items || data.items.length === 0) {
      console.error(`No book found for ISBN ${isbn}`);
      return null;
    }

    const book = data.items[0].volumeInfo;

    return {
      originalTitle: book.title,
      year: book.publishedDate?.substring(0, 4),
      author: book.authors?.join(', '),
      genre: book.categories || [],
      pages: book.pageCount ? `${book.pageCount}쪽` : '',
      posterUrl: book.imageLinks?.thumbnail?.replace('http:', 'https:'),
    };
  } catch (error) {
    console.error(`Failed to fetch Google Books metadata for ISBN ${isbn}:`, error);
    return null;
  }
}

async function downloadImage(url: string, outputPath: string): Promise<void> {
  try {
    const response = await axios.get(url, { responseType: 'arraybuffer' });
    const buffer = Buffer.from(response.data);

    // Optimize and resize image
    await sharp(buffer)
      .resize(500, null, { withoutEnlargement: true })
      .jpeg({ quality: 90 })
      .toFile(outputPath);

    console.log(`Downloaded and optimized image: ${outputPath}`);
  } catch (error) {
    console.error(`Failed to download image from ${url}:`, error);
  }
}

function removeUndefinedValues(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(removeUndefinedValues).filter(v => v !== undefined);
  }

  if (obj !== null && typeof obj === 'object') {
    const cleaned: any = {};
    for (const [key, value] of Object.entries(obj)) {
      if (value !== undefined) {
        const cleanedValue = removeUndefinedValues(value);
        // Don't include empty objects or empty arrays
        if (typeof cleanedValue === 'object' && cleanedValue !== null) {
          if (Array.isArray(cleanedValue) && cleanedValue.length === 0) continue;
          if (!Array.isArray(cleanedValue) && Object.keys(cleanedValue).length === 0) continue;
        }
        cleaned[key] = cleanedValue;
      }
    }
    return cleaned;
  }

  return obj;
}

async function processReviewFile(filePath: string) {
  const fileContent = fs.readFileSync(filePath, 'utf-8');
  // Parse with engines option to keep dates as strings
  const parsed = matter(fileContent, {
    engines: {
      yaml: {
        parse: (str: string) => {
          const yaml = require('js-yaml');
          return yaml.load(str, { schema: yaml.JSON_SCHEMA });
        }
      }
    }
  });
  const frontmatter = parsed.data as ReviewFrontmatter;
  const content = parsed.content;

  if (!frontmatter.externalIds) {
    console.log(`Skipping ${path.basename(filePath)}: No externalIds found`);
    return;
  }

  let metadata: any = frontmatter.metadata || {};
  let posterUrl: string | null = null;
  let updated = false;

  // Fetch metadata based on external IDs
  if (frontmatter.externalIds.tmdbId && frontmatter.mediaType !== 'book') {
    if (!TMDB_API_KEY) {
      console.log(`Skipping ${path.basename(filePath)}: TMDB API key not set`);
      return;
    }

    const tmdbData = await fetchTMDBMetadata(
      frontmatter.externalIds.tmdbId,
      frontmatter.mediaType as 'movie' | 'series' | 'animation'
    );

    if (tmdbData) {
      metadata = {
        ...metadata,
        originalTitle: metadata.originalTitle || tmdbData.originalTitle,
        year: metadata.year || tmdbData.year,
        director: metadata.director || tmdbData.director,
        creator: metadata.creator || tmdbData.creator,
        genre: metadata.genre || tmdbData.genre,
        runtime: metadata.runtime || tmdbData.runtime,
      };
      posterUrl = tmdbData.posterUrl;
      updated = true;
    }
  } else if (frontmatter.externalIds.isbn) {
    if (!GOOGLE_BOOKS_API_KEY) {
      console.log(`⚠️  ${path.basename(filePath)}: Google Books API key not set (will use existing metadata if available)`);
    } else {
      const bookData = await fetchGoogleBooksMetadata(frontmatter.externalIds.isbn);

      if (bookData) {
        metadata = {
          ...metadata,
          originalTitle: metadata.originalTitle || bookData.originalTitle,
          year: metadata.year || bookData.year,
          author: metadata.author || bookData.author,
          genre: metadata.genre || bookData.genre,
          pages: metadata.pages || bookData.pages,
        };
        posterUrl = bookData.posterUrl;
        updated = true;
      }
    }
  }

  // Download poster image if available and not already downloaded
  const reviewDir = path.dirname(filePath);
  const posterPath = path.join(reviewDir, 'poster.jpg');

  if (posterUrl) {
    if (!fs.existsSync(posterPath)) {
      await downloadImage(posterUrl, posterPath);
      frontmatter.poster = './poster.jpg';
      updated = true;
    }
  } else {
    // Check if poster was manually added
    if (fs.existsSync(posterPath) && !frontmatter.poster) {
      frontmatter.poster = './poster.jpg';
      updated = true;
      console.log(`📷 Found manual poster for ${path.basename(filePath)}`);
    }
  }

  // Update frontmatter if changes were made
  if (updated) {
    frontmatter.metadata = metadata;

    // Remove undefined values before stringifying
    const cleanedFrontmatter = removeUndefinedValues(frontmatter);
    const updatedContent = matter.stringify(content, cleanedFrontmatter);
    fs.writeFileSync(filePath, updatedContent, 'utf-8');

    console.log(`Updated ${path.basename(filePath)}`);
  } else {
    console.log(`No updates for ${path.basename(filePath)}`);
  }
}

async function main() {
  if (!fs.existsSync(REVIEWS_DIR)) {
    console.error(`Reviews directory not found: ${REVIEWS_DIR}`);
    process.exit(1);
  }

  // Show API key status
  console.log('API Key Status:');
  console.log(`- TMDB API Key: ${TMDB_API_KEY ? '✓ Set' : '✗ Not set'}`);
  console.log(`- Google Books API Key: ${GOOGLE_BOOKS_API_KEY ? '✓ Set' : '✗ Not set'}`);
  console.log('');

  if (!TMDB_API_KEY && !GOOGLE_BOOKS_API_KEY) {
    console.warn('Warning: No API keys found. Metadata fetching will be skipped.');
    console.warn('Set TMDB_API_KEY for movies/series/animation or GOOGLE_BOOKS_API_KEY for books.');
  }

  // Find all index.md files in review folders
  const folders = fs.readdirSync(REVIEWS_DIR)
    .filter(item => {
      const itemPath = path.join(REVIEWS_DIR, item);
      return fs.statSync(itemPath).isDirectory();
    });

  const files = folders
    .map(folder => path.join(REVIEWS_DIR, folder, 'index.md'))
    .filter(filePath => fs.existsSync(filePath));

  console.log(`Found ${files.length} review files`);

  for (const file of files) {
    await processReviewFile(file);
  }

  console.log('\nDone!');
}

main().catch(console.error);
